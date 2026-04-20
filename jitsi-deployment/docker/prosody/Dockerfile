ARG JITSI_IMAGE_VERSION=stable-10741
FROM jitsi/prosody:${JITSI_IMAGE_VERSION}

COPY assets/meeting_portal_app/prosody-init/20-meeting-portal-post-config /etc/cont-init.d/20-meeting-portal-post-config

RUN chmod 755 /etc/cont-init.d/20-meeting-portal-post-config
