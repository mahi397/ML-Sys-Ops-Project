#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ONLINE_INFERENCE_WORKFLOW_ENV_FILE:-${SCRIPT_DIR}/online_inference_workflow.env}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATABASE_URL="${DATABASE_URL:-postgresql://proj07_user:proj07@127.0.0.1:5432/proj07_sql_db}"
RCLONE_REMOTE="${RCLONE_REMOTE:-rclone_s3}"
BUCKET="${BUCKET:-objstore-proj07}"

TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-/mnt/block/user-behaviour/Transcripts}"
PARSED_TRANSCRIPT_ROOT="${PARSED_TRANSCRIPT_ROOT:-/mnt/block/user-behaviour/parsed_transcripts}"
ONLINE_STAGE1_ROOT="${ONLINE_STAGE1_ROOT:-/mnt/block/user-behaviour/online_inference/stage1}"
ONLINE_STAGE2_ROOT="${ONLINE_STAGE2_ROOT:-/mnt/block/user-behaviour/online_inference/stage2}"
STAGE1_RESPONSE_ROOT="${STAGE1_RESPONSE_ROOT:-/mnt/block/user-behaviour/inference_responses/stage1}"
STAGE2_RESPONSE_ROOT="${STAGE2_RESPONSE_ROOT:-/mnt/block/user-behaviour/inference_responses/stage2}"
SEGMENTS_ROOT="${SEGMENTS_ROOT:-/mnt/block/user-behaviour/reconstructed_segments}"
RECAP_ROOT="${RECAP_ROOT:-/mnt/block/user-behaviour/recaps/generated}"
LOG_ROOT="${LOG_ROOT:-/mnt/block/user-behaviour/logs/online_inference_workflow}"

STAGE1_MODE="${STAGE1_MODE:-mock}"
STAGE2_MODE="${STAGE2_MODE:-mock}"
VERSION="${VERSION:-1}"
FEEDBACK_VERSION="${FEEDBACK_VERSION:-${VERSION}}"
STRUCTURAL_EVENT="${STRUCTURAL_EVENT:-auto}"
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-/mnt/block/staging/feedback_loop}"
FEEDBACK_PREFIX="${FEEDBACK_PREFIX:-production}"
WINDOW_SIZE="${WINDOW_SIZE:-7}"
TRANSITION_INDEX="${TRANSITION_INDEX:-3}"
MIN_UTTERANCE_CHARS="${MIN_UTTERANCE_CHARS:-1}"
MAX_WORDS_PER_UTTERANCE="${MAX_WORDS_PER_UTTERANCE:-50}"

export DATABASE_URL
export RCLONE_REMOTE
export BUCKET
export LOCAL_TMP_ROOT
export FEEDBACK_PREFIX
export WINDOW_SIZE
export TRANSITION_INDEX
export MIN_UTTERANCE_CHARS
export MAX_WORDS_PER_UTTERANCE

mkdir -p \
  "${TRANSCRIPT_ROOT}" \
  "${PARSED_TRANSCRIPT_ROOT}" \
  "${ONLINE_STAGE1_ROOT}" \
  "${ONLINE_STAGE2_ROOT}" \
  "${STAGE1_RESPONSE_ROOT}" \
  "${STAGE2_RESPONSE_ROOT}" \
  "${SEGMENTS_ROOT}" \
  "${RECAP_ROOT}" \
  "${LOCAL_TMP_ROOT}" \
  "${LOG_ROOT}"

function derive_meeting_id_from_filename() {
  local transcript_filename
  transcript_filename="$(basename "$1")"

  if [[ ! "${transcript_filename}" =~ ^transcript_([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(\.[0-9]+)?Z_([0-9a-fA-F]{8})[0-9a-fA-F-]*\.txt$ ]]; then
    echo "Invalid Jitsi transcript filename: ${transcript_filename}" >&2
    exit 1
  fi

  printf 'jitsi_%s%s%sT%s%s%sZ_%s\n' \
    "${BASH_REMATCH[1]}" \
    "${BASH_REMATCH[2]}" \
    "${BASH_REMATCH[3]}" \
    "${BASH_REMATCH[4]}" \
    "${BASH_REMATCH[5]}" \
    "${BASH_REMATCH[6]}" \
    "${BASH_REMATCH[8],,}"
}

function process_transcript() {
  local transcript_path="$1"
  local transcript_filename
  local meeting_id

  transcript_filename="$(basename "${transcript_path}")"
  meeting_id="$(derive_meeting_id_from_filename "${transcript_filename}")"

  echo
  echo "=== Online inference workflow transcript ==="
  echo "Transcript: ${transcript_filename}"
  echo "Meeting ID: ${meeting_id}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/ingest_jitsi_transcript.py" \
    --transcript-path "${transcript_path}" \
    --replace-existing \
    --build-stage1-after-ingest \
    --upload-stage1-artifacts \
    --version "${VERSION}" \
    --local-output-root "${PARSED_TRANSCRIPT_ROOT}" \
    --stage1-output-root "${ONLINE_STAGE1_ROOT}" \
    --stage1-object-prefix "production/inference_requests/stage1" \
    --log-file "${LOG_ROOT}/${meeting_id}_ingest.log"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/replay_to_hypothetical_endpoints.py" \
    --meeting-id "${meeting_id}" \
    --version "${VERSION}" \
    --stage1-mode "${STAGE1_MODE}" \
    --stage2-mode "${STAGE2_MODE}" \
    --upload-artifacts \
    --stage1-response-root "${STAGE1_RESPONSE_ROOT}" \
    --stage2-input-root "${ONLINE_STAGE2_ROOT}" \
    --stage2-response-root "${STAGE2_RESPONSE_ROOT}" \
    --segments-root "${SEGMENTS_ROOT}" \
    --recap-root "${RECAP_ROOT}" \
    --rclone-remote "${RCLONE_REMOTE}" \
    --bucket "${BUCKET}" \
    --log-file "${LOG_ROOT}/${meeting_id}_replay.log"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_feedback_events.py" \
    --meeting-id "${meeting_id}" \
    --version "${FEEDBACK_VERSION}" \
    --structural-event "${STRUCTURAL_EVENT}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/materialize_corrected_recap.py" \
    --meeting-id "${meeting_id}" \
    --version "${FEEDBACK_VERSION}"
}

declare -a transcript_paths=()
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    if [[ "${arg}" = /* ]]; then
      transcript_paths+=("${arg}")
    else
      transcript_paths+=("${TRANSCRIPT_ROOT}/${arg}")
    fi
  done
else
  while IFS= read -r line; do
    transcript_paths+=("${line}")
  done < <(find "${TRANSCRIPT_ROOT}" -maxdepth 1 -type f -name 'transcript_*.txt' | sort)
fi

if [[ ${#transcript_paths[@]} -eq 0 ]]; then
  echo "No transcript_*.txt files found under ${TRANSCRIPT_ROOT}" >&2
  exit 1
fi

for transcript_path in "${transcript_paths[@]}"; do
  if [[ ! -f "${transcript_path}" ]]; then
    echo "Transcript not found: ${transcript_path}" >&2
    exit 1
  fi
  process_transcript "${transcript_path}"
done

echo
echo "Online inference workflow runtime completed for ${#transcript_paths[@]} transcript(s)."
