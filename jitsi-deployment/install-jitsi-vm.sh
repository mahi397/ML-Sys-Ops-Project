#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

INSTALL_ROOT="/mnt/block/jitsi"
ENV_SOURCE=""
PROJECT_NAME="jitsi-vm"
SKIP_DOCKER_INSTALL=0
SKIP_VOSK_MODEL_DOWNLOAD=0

usage() {
    cat <<'EOF'
Usage:
  bash install-jitsi-vm.sh [options]

Options:
  --env-file PATH         Deployment env file to merge into .env
  --install-root PATH     Persistent install root (default: /mnt/block/jitsi)
  --project-name NAME     Docker Compose project name (default: jitsi-vm)
  --skip-docker-install   Do not install Docker automatically
  --skip-vosk-download    Do not download the Vosk model automatically
  --help                  Show this help text
EOF
}

log() {
    printf '[jitsi-deploy] %s\n' "$*"
}

warn() {
    printf '[jitsi-deploy] WARNING: %s\n' "$*" >&2
}

fatal() {
    printf '[jitsi-deploy] ERROR: %s\n' "$*" >&2
    exit 1
}

fetch_text_url() {
    local url="$1"
    local description="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL \
            -H 'Accept: application/vnd.github+json' \
            -H 'User-Agent: jitsi-deploy-installer' \
            "$url" || fatal "Failed to ${description} from ${url}. Check outbound HTTPS access from the VM."
        return 0
    fi

    wget -q -O - \
        --header='Accept: application/vnd.github+json' \
        --user-agent='jitsi-deploy-installer' \
        "$url" || fatal "Failed to ${description} from ${url}. Check outbound HTTPS access from the VM."
}

download_file() {
    local url="$1"
    local destination="$2"
    local description="$3"

    if command -v curl >/dev/null 2>&1; then
        curl -fL "$url" -o "$destination" || fatal "Failed to ${description} from ${url}."
        return 0
    fi

    wget -O "$destination" "$url" || fatal "Failed to ${description} from ${url}."
}

strip_wrapping_quotes() {
    local value="$1"

    if [[ "$value" =~ ^\".*\"$ ]]; then
        value="${value:1:${#value}-2}"
    elif [[ "$value" =~ ^\'.*\'$ ]]; then
        value="${value:1:${#value}-2}"
    fi

    printf '%s\n' "$value"
}

read_env_value() {
    local key="$1"
    local file="$2"

    awk -F= -v key="$key" '
        $1 == key {
            value = substr($0, index($0, "=") + 1)
        }
        END {
            print value
        }
    ' "$file"
}

set_env_value() {
    local key="$1"
    local value="$2"
    local file="$3"
    local tmp

    tmp="$(mktemp)"
    awk -v key="$key" -v value="$value" '
        BEGIN {
            done = 0
        }
        $0 ~ ("^[[:space:]]*#?[[:space:]]*" key "=") {
            if (!done) {
                print key "=" value
                done = 1
            }
            next
        }
        {
            print
        }
        END {
            if (!done) {
                print key "=" value
            }
        }
    ' "$file" > "$tmp"
    mv "$tmp" "$file"
}

merge_env_file() {
    local src="$1"
    local dst="$2"
    local line key value

    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|\#*)
                continue
                ;;
        esac

        key="${line%%=*}"
        value="${line#*=}"

        if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            continue
        fi

        if [ -n "$value" ]; then
            set_env_value "$key" "$value" "$dst"
        fi
    done < "$src"
}

value_is_blank_or_placeholder() {
    local value="$1"

    value="$(strip_wrapping_quotes "$value")"

    case "$value" in
        ''|change-me-jitsi-jwt-secret|replace-me|replace_with_real_value|your_actual_token_here|\"your_actual_token_here\"|example|changeme)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

generate_if_needed() {
    local key="$1"
    local file="$2"
    local current generator_value

    current="$(read_env_value "$key" "$file")"
    if ! value_is_blank_or_placeholder "$current"; then
        return 0
    fi

    case "$key" in
        MEETING_PORTAL_SESSION_SECRET|JWT_APP_SECRET|INGEST_TOKEN)
            generator_value="$(openssl rand -hex 32)"
            ;;
        JITSI_HOST_EXTERNAL_KEY)
            if command -v uuidgen >/dev/null 2>&1; then
                generator_value="$(uuidgen)"
            else
                generator_value="$(cat /proc/sys/kernel/random/uuid)"
            fi
            ;;
        *)
            fatal "Unsupported generated key: $key"
            ;;
    esac

    set_env_value "$key" "$generator_value" "$file"
}

require_env_value() {
    local key="$1"
    local value

    value="$(read_env_value "$key" "$ENV_TARGET")"
    if value_is_blank_or_placeholder "$value"; then
        fatal "Missing required .env value: $key"
    fi
}

require_env_regex() {
    local key="$1"
    local regex="$2"
    local description="$3"
    local value

    value="$(strip_wrapping_quotes "$(read_env_value "$key" "$ENV_TARGET")")"
    require_env_value "$key"

    if [[ ! "$value" =~ $regex ]]; then
        fatal "Invalid .env value for $key: expected $description, got '$value'"
    fi
}

require_env_one_of() {
    local key="$1"
    shift
    local value candidate

    value="$(strip_wrapping_quotes "$(read_env_value "$key" "$ENV_TARGET")")"
    require_env_value "$key"

    for candidate in "$@"; do
        if [ "$value" = "$candidate" ]; then
            return 0
        fi
    done

    fatal "Invalid .env value for $key: got '$value', expected one of: $*"
}

require_env_boolean() {
    local key="$1"

    require_env_regex "$key" '^(1|0|true|false|yes|no|on|off)$' "a boolean value"
}

env_key_is_truthy() {
    local key="$1"
    local file="$2"
    local value

    value="$(strip_wrapping_quotes "$(read_env_value "$key" "$file")")"

    case "$value" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

require_env_integer() {
    local key="$1"

    require_env_regex "$key" '^[0-9]+$' "a non-negative integer"
}

require_env_positive_number() {
    local key="$1"

    require_env_regex "$key" '^[0-9]+([.][0-9]+)?$' "a positive number"
}

require_env_url() {
    local key="$1"
    local scheme_regex="$2"

    require_env_regex "$key" "^(${scheme_regex})://[^[:space:]]+$" "a URL matching ${scheme_regex}://..."
}

require_env_path() {
    local key="$1"

    require_env_regex "$key" '^/[^[:space:]]*$' "an absolute path"
}

require_demo_value_replaced() {
    local key="$1"
    local bad_value="$2"
    local message="$3"
    local value

    value="$(strip_wrapping_quotes "$(read_env_value "$key" "$ENV_TARGET")")"
    require_env_value "$key"

    if [ "$value" = "$bad_value" ]; then
        fatal "$message"
    fi
}

validate_auth_and_prosody_env() {
    require_env_url PUBLIC_URL 'https?'
    require_env_regex JVB_ADVERTISE_IPS '^[^[:space:]]+$' "an IP or comma-separated IP list without spaces"

    require_env_one_of ENABLE_AUTH 1 true yes on
    require_env_one_of ENABLE_GUESTS 1 true yes on
    require_env_one_of AUTH_TYPE jwt

    require_env_value JWT_APP_ID
    require_env_value JWT_APP_SECRET
    require_env_value JWT_ACCEPTED_ISSUERS
    require_env_value JWT_ACCEPTED_AUDIENCES
    require_env_one_of JWT_ALLOW_EMPTY 0 false no off
    require_env_regex TOKEN_AUTH_URL '^/[^[:space:]]*$' "an absolute URL path beginning with /"
    require_env_one_of JICOFO_ENABLE_AUTH 0 false no off
    require_env_one_of WAIT_FOR_HOST_DISABLE_AUTO_OWNERS 1 true yes on

    require_env_value XMPP_DOMAIN
    require_env_value XMPP_SERVER
    require_env_url XMPP_BOSH_URL_BASE 'https?|ws|wss'
    require_env_value XMPP_AUTH_DOMAIN
    require_env_value XMPP_GUEST_DOMAIN
    require_env_value XMPP_MUC_DOMAIN
    require_env_value XMPP_INTERNAL_MUC_DOMAIN
    require_env_value XMPP_HIDDEN_DOMAIN

    require_env_value JICOFO_AUTH_PASSWORD
    require_env_value JVB_AUTH_PASSWORD
    require_env_value JIGASI_XMPP_PASSWORD
    require_env_value JIGASI_TRANSCRIBER_PASSWORD
}

validate_jigasi_env() {
    local disable_sip

    disable_sip="$(strip_wrapping_quotes "$(read_env_value JIGASI_DISABLE_SIP "$ENV_TARGET")")"
    if [ -n "$disable_sip" ]; then
        require_env_boolean JIGASI_DISABLE_SIP
    fi

    if env_key_is_truthy JIGASI_DISABLE_SIP "$ENV_TARGET"; then
        return 0
    fi

    require_env_value JIGASI_SIP_URI
    require_env_value JIGASI_SIP_PASSWORD
    require_env_value JIGASI_SIP_SERVER
    require_env_integer JIGASI_SIP_PORT
    require_env_one_of JIGASI_SIP_TRANSPORT UDP TCP TLS

    require_demo_value_replaced JIGASI_SIP_URI 'test@sip2sip.info' "JIGASI_SIP_URI is still using the demo placeholder value."
    require_demo_value_replaced JIGASI_SIP_PASSWORD 'passw0rd' "JIGASI_SIP_PASSWORD is still using the demo placeholder value."
    require_demo_value_replaced JIGASI_SIP_SERVER 'sip2sip.info' "JIGASI_SIP_SERVER is still using the demo placeholder value."
}

validate_transcription_and_vosk_env() {
    require_env_one_of ENABLE_TRANSCRIPTIONS 1 true yes on
    require_env_one_of ENABLE_P2P 0 false no off
    require_env_one_of JIGASI_TRANSCRIBER_ENABLE_SAVING 1 true yes on
    require_env_boolean JIGASI_TRANSCRIBER_RECORD_AUDIO
    require_env_boolean JIGASI_TRANSCRIBER_FILTER_SILENCE
    require_env_one_of JIGASI_TRANSCRIBER_CUSTOM_SERVICE org.jitsi.jigasi.transcription.VoskTranscriptionService
    require_env_url JIGASI_TRANSCRIBER_VOSK_URL 'ws|wss'
    require_env_url VOSK_MODEL_URL 'https?'
    require_env_path VOSK_MODEL_PATH
}

validate_transcript_uploader_env() {
    require_env_url JITSI_TRANSCRIPT_INGEST_URL 'https?'
    require_env_value INGEST_TOKEN
    require_env_regex JITSI_HOST_EXTERNAL_KEY '^[0-9A-Fa-f-]{36}$' "a UUID such as 123e4567-e89b-12d3-a456-426614174000"
    require_env_positive_number JITSI_TRANSCRIPT_POLL_SECONDS
    require_env_positive_number JITSI_TRANSCRIPT_SETTLE_SECONDS
    require_env_positive_number JITSI_TRANSCRIPT_UPLOAD_TIMEOUT
}

validate_meeting_portal_env() {
    require_env_regex MEETING_PORTAL_DATABASE_URL '^(postgres|postgresql)://[^[:space:]]+$' "a PostgreSQL DSN"
    require_env_value MEETING_PORTAL_SESSION_SECRET
    require_env_boolean MEETING_PORTAL_HTTPS_ONLY
    require_env_integer MEETING_PORTAL_TOKEN_TTL_SECONDS
    require_env_value MEETING_PORTAL_RCLONE_REMOTE
    require_env_value MEETING_PORTAL_RCLONE_BUCKET
    require_env_integer MEETING_PORTAL_RCLONE_TIMEOUT_SECONDS
    require_env_boolean MEETING_PORTAL_STAGE1_RCLONE_FALLBACK_ENABLED
}

detect_primary_ipv4() {
    local ip_address

    ip_address="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}')"
    if [ -n "$ip_address" ]; then
        printf '%s\n' "$ip_address"
        return 0
    fi

    ip_address="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -n "$ip_address" ]; then
        printf '%s\n' "$ip_address"
        return 0
    fi

    return 1
}

ensure_base_packages() {
    if ! command -v apt-get >/dev/null 2>&1; then
        fatal "Automatic package installation is only implemented for apt-based systems."
    fi

    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y ca-certificates curl gpg openssl rsync unzip uuid-runtime wget
}

install_docker_if_needed() {
    local os_id version_codename repo_url gpg_url docker_ok compose_ok

    if [ "$SKIP_DOCKER_INSTALL" -eq 1 ]; then
        return 0
    fi

    docker_ok=0
    compose_ok=0

    if command -v docker >/dev/null 2>&1; then
        docker_ok=1
        if docker compose version >/dev/null 2>&1; then
            compose_ok=1
        fi
    fi

    if [ "$docker_ok" -eq 1 ] && [ "$compose_ok" -eq 1 ]; then
        return 0
    fi

    log "Installing Docker Engine and Docker Compose plugin"

    if [ ! -f /etc/os-release ]; then
        fatal "Cannot determine Linux distribution."
    fi

    # shellcheck disable=SC1091
    . /etc/os-release
    os_id="${ID}"
    version_codename="${VERSION_CODENAME:-}"

    case "$os_id" in
        ubuntu)
            version_codename="${UBUNTU_CODENAME:-$version_codename}"
            ;;
        debian)
            ;;
        *)
            fatal "Automatic Docker installation is only implemented for Ubuntu and Debian."
            ;;
    esac

    [ -n "$version_codename" ] || fatal "Could not determine the distro codename for Docker apt repository setup."

    repo_url="https://download.docker.com/linux/${os_id}"
    gpg_url="${repo_url}/gpg"

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL "$gpg_url" -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc

    cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: ${repo_url}
Suites: ${version_codename}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

    apt-get update
    if ! apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin; then
        warn "Docker package install failed once; retrying after removing conflicting distro packages"
        apt-get remove -y docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc || true
        apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    fi

    if command -v systemctl >/dev/null 2>&1; then
        systemctl enable --now docker
    fi
}

create_directories() {
    local config_root="$1"

    mkdir -p \
        "$INSTALL_ROOT" \
        "$SOURCE_ROOT" \
        "$config_root/web/crontabs" \
        "$config_root/web/load-test" \
        "$config_root/transcripts" \
        "$config_root/prosody/config" \
        "$config_root/prosody/prosody-plugins-custom" \
        "$config_root/jicofo" \
        "$config_root/jvb" \
        "$config_root/jigasi" \
        "$config_root/jibri" \
        "$config_root/transcriber" \
        "$config_root/meeting-portal-app/room-contexts" \
        "$config_root/transcript-uploader" \
        "$config_root/rclone" \
        "$config_root/models"
}

download_upstream_source() {
    local api_url zip_url download_dir archive_path extract_dir extracted_source metadata

    api_url="https://api.github.com/repos/jitsi/docker-jitsi-meet/releases/latest"
    download_dir="${INSTALL_ROOT}/downloads"
    archive_path="${download_dir}/docker-jitsi-meet-latest.zip"
    extract_dir="${download_dir}/docker-jitsi-meet-release"

    mkdir -p "$download_dir"

    log "Fetching latest docker-jitsi-meet release metadata"
    metadata="$(fetch_text_url "$api_url" "fetch docker-jitsi-meet release metadata")"
    zip_url="$(printf '%s\n' "$metadata" | sed -n 's/.*"zipball_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
    [ -n "$zip_url" ] || fatal "Could not determine the latest docker-jitsi-meet release zip URL."

    log "Downloading upstream Jitsi source"
    download_file "$zip_url" "$archive_path" "download the upstream Jitsi source archive"

    rm -rf "$extract_dir"
    mkdir -p "$extract_dir"
    unzip -q -o "$archive_path" -d "$extract_dir"

    extracted_source="$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [ -n "$extracted_source" ] || fatal "Failed to extract the upstream docker-jitsi-meet release."

    log "Syncing upstream Jitsi source into ${SOURCE_ROOT}"
    rsync -a \
        --delete \
        --exclude '.env' \
        --exclude '.env.bak' \
        "$extracted_source"/ "${SOURCE_ROOT}/"
}

sync_deployment_bundle() {
    log "Syncing deployment bundle into ${BUNDLE_ROOT}"

    mkdir -p "$BUNDLE_ROOT"
    rsync -a \
        --delete \
        --exclude '.git/' \
        --exclude 'stack.env' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        "${SCRIPT_DIR}/" "${BUNDLE_ROOT}/"
}

normalize_script_permissions() {
    log "Normalizing script permissions in the synced source tree"

    find "$SOURCE_ROOT" -type f \
        \( \
            -path '*/etc/services.d/*/run' -o \
            -path '*/etc/services.d/*/finish' -o \
            -path '*/etc/cont-init.d/*' -o \
            -path '*/prosody-init/*' -o \
            -name '*.sh' \
        \) \
        -exec chmod 755 {} +
}

ensure_env_source() {
    local default_env_source

    default_env_source="${SCRIPT_DIR}/stack.env"

    if [ -n "$ENV_SOURCE" ]; then
        [ -f "$ENV_SOURCE" ] || fatal "Env file not found: $ENV_SOURCE"
        return 0
    fi

    if [ -f "$default_env_source" ]; then
        ENV_SOURCE="$default_env_source"
        return 0
    fi

    cp "${SCRIPT_DIR}/stack.env.example" "$default_env_source"
    fatal "Created ${default_env_source}. Fill in the required values and rerun the installer."
}

ensure_env_target() {
    if [ ! -f "$ENV_TARGET" ]; then
        cp "${SOURCE_ROOT}/env.example" "$ENV_TARGET"
    fi
}

set_inferred_network_values() {
    local public_url advertised_ips public_host inferred_ip

    public_url="$(read_env_value PUBLIC_URL "$ENV_TARGET")"
    advertised_ips="$(read_env_value JVB_ADVERTISE_IPS "$ENV_TARGET")"
    inferred_ip="$(detect_primary_ipv4 || true)"

    if [ -z "$public_url" ] && [ -n "$inferred_ip" ]; then
        set_env_value PUBLIC_URL "https://${inferred_ip}" "$ENV_TARGET"
        public_url="https://${inferred_ip}"
    fi

    if [ -z "$advertised_ips" ]; then
        public_host="${public_url#http://}"
        public_host="${public_host#https://}"
        public_host="${public_host%%/*}"
        public_host="${public_host%%:*}"

        if [[ "$public_host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            set_env_value JVB_ADVERTISE_IPS "$public_host" "$ENV_TARGET"
        elif [ -n "$inferred_ip" ]; then
            set_env_value JVB_ADVERTISE_IPS "$inferred_ip" "$ENV_TARGET"
        fi
    fi
}

set_vosk_paths() {
    local model_url model_zip model_dir model_path

    model_url="$(read_env_value VOSK_MODEL_URL "$ENV_TARGET")"
    if [ -z "$model_url" ]; then
        model_url="https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
        set_env_value VOSK_MODEL_URL "$model_url" "$ENV_TARGET"
    fi

    model_zip="${model_url##*/}"
    model_dir="${model_zip%.zip}"
    model_path="${CONFIG_ROOT}/models/${model_dir}"
    set_env_value VOSK_MODEL_PATH "$model_path" "$ENV_TARGET"
}

generate_jitsi_passwords_if_needed() {
    local keys key value missing

    missing=0
    keys=(
        JICOFO_AUTH_PASSWORD
        JVB_AUTH_PASSWORD
        JIGASI_XMPP_PASSWORD
        JIBRI_RECORDER_PASSWORD
        JIBRI_XMPP_PASSWORD
        JIGASI_TRANSCRIBER_PASSWORD
    )

    for key in "${keys[@]}"; do
        value="$(read_env_value "$key" "$ENV_TARGET")"
        if [ -z "$value" ]; then
            missing=1
            break
        fi
    done

    if [ "$missing" -eq 1 ]; then
        log "Generating Jitsi internal service passwords"
        bash "${SOURCE_ROOT}/gen-passwords.sh"
    fi
}

validate_env() {
    validate_auth_and_prosody_env
    validate_jigasi_env
    validate_transcription_and_vosk_env
    validate_transcript_uploader_env
    validate_meeting_portal_env

    if [ "$(read_env_value ENABLE_LETSENCRYPT "$ENV_TARGET")" = "1" ]; then
        require_env_value LETSENCRYPT_DOMAIN
        require_env_regex LETSENCRYPT_EMAIL '^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$' "an email address"
    fi
}

prepare_env() {
    ensure_env_source
    ensure_env_target
    merge_env_file "$ENV_SOURCE" "$ENV_TARGET"

    set_env_value CONFIG "$CONFIG_ROOT" "$ENV_TARGET"
    if [ -z "$(strip_wrapping_quotes "$(read_env_value JIGASI_DISABLE_SIP "$ENV_TARGET")")" ]; then
        set_env_value JIGASI_DISABLE_SIP 1 "$ENV_TARGET"
    fi
    set_inferred_network_values
    set_vosk_paths

    generate_if_needed MEETING_PORTAL_SESSION_SECRET "$ENV_TARGET"
    generate_if_needed JWT_APP_SECRET "$ENV_TARGET"
    generate_if_needed INGEST_TOKEN "$ENV_TARGET"
    generate_if_needed JITSI_HOST_EXTERNAL_KEY "$ENV_TARGET"
    generate_jitsi_passwords_if_needed

    validate_env
}

download_vosk_model() {
    local model_url model_path temp_dir extracted_dir parent_dir

    if [ "$SKIP_VOSK_MODEL_DOWNLOAD" -eq 1 ]; then
        return 0
    fi

    model_url="$(read_env_value VOSK_MODEL_URL "$ENV_TARGET")"
    model_path="$(read_env_value VOSK_MODEL_PATH "$ENV_TARGET")"
    parent_dir="$(dirname "$model_path")"

    if [ -d "$model_path" ] && [ -n "$(find "$model_path" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
        return 0
    fi

    log "Downloading Vosk model into ${model_path}"

    mkdir -p "$parent_dir"
    temp_dir="$(mktemp -d)"
    download_file "$model_url" "${temp_dir}/model.zip" "download the Vosk model archive"
    unzip -q "${temp_dir}/model.zip" -d "$temp_dir"
    extracted_dir="$(find "$temp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [ -n "$extracted_dir" ] || fatal "Failed to extract the Vosk model archive."
    rm -rf "$model_path"
    mv "$extracted_dir" "$model_path"
    rm -rf "$temp_dir"
}

warn_if_optional_files_missing() {
    if [ ! -f "${CONFIG_ROOT}/rclone/rclone.conf" ]; then
        warn "No rclone config found at ${CONFIG_ROOT}/rclone/rclone.conf. Meeting portal recap artifact fetches that depend on rclone will fail until it is added."
    fi
}

run_compose() {
    local compose_args

    compose_args=(
        --project-name "$PROJECT_NAME"
        -f docker-compose.yml
        -f jigasi.yml
        -f transcriber.yml
        -f jitsi-deployment/compose/vm-services.yml
    )

    log "Validating Docker Compose configuration"
    (
        cd "$SOURCE_ROOT"
        docker compose "${compose_args[@]}" config >/dev/null
    )

    log "Building and starting the Jitsi stack"
    (
        cd "$SOURCE_ROOT"
        docker compose "${compose_args[@]}" up -d --build --remove-orphans
        docker compose "${compose_args[@]}" ps
    )
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --env-file)
                [ "$#" -ge 2 ] || fatal "--env-file requires a path"
                ENV_SOURCE="$2"
                shift 2
                ;;
            --install-root)
                [ "$#" -ge 2 ] || fatal "--install-root requires a path"
                INSTALL_ROOT="$2"
                shift 2
                ;;
            --project-name)
                [ "$#" -ge 2 ] || fatal "--project-name requires a value"
                PROJECT_NAME="$2"
                shift 2
                ;;
            --skip-docker-install)
                SKIP_DOCKER_INSTALL=1
                shift
                ;;
            --skip-vosk-download)
                SKIP_VOSK_MODEL_DOWNLOAD=1
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                fatal "Unknown option: $1"
                ;;
        esac
    done
}

main() {
    parse_args "$@"

    if [ "$(uname -s)" != "Linux" ]; then
        fatal "This installer targets Linux VMs only."
    fi

    if [ "${EUID}" -ne 0 ]; then
        exec sudo bash "$0" "$@"
    fi

    if [ ! -d /mnt/block ] && [ "$INSTALL_ROOT" = "/mnt/block/jitsi" ]; then
        fatal "Persistent storage mount /mnt/block was not found."
    fi

    SOURCE_ROOT="${INSTALL_ROOT}/jitsi-docker-jitsi-meet"
    BUNDLE_ROOT="${SOURCE_ROOT}/jitsi-deployment"
    CONFIG_ROOT="${INSTALL_ROOT}/config"
    ENV_TARGET="${SOURCE_ROOT}/.env"

    ensure_base_packages
    install_docker_if_needed
    if command -v systemctl >/dev/null 2>&1; then
        systemctl is-active --quiet docker || systemctl start docker || true
    fi
    create_directories "$CONFIG_ROOT"
    download_upstream_source
    sync_deployment_bundle
    normalize_script_permissions
    prepare_env
    download_vosk_model
    warn_if_optional_files_missing
    run_compose

    log "Deployment completed."
    log "Source root: ${SOURCE_ROOT}"
    log "Config root: ${CONFIG_ROOT}"
    log "Open: $(read_env_value PUBLIC_URL "$ENV_TARGET")"
}

main "$@"
