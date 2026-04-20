ARG JITSI_IMAGE_VERSION=stable-10741
FROM jitsi/web:${JITSI_IMAGE_VERSION}

COPY assets/meeting_portal_app/jitsi-web/title.html /usr/share/jitsi-meet/title.html
COPY assets/meeting_portal_app/jitsi-web/plugin.head.html /usr/share/jitsi-meet/plugin.head.html
COPY assets/meeting_portal_app/jitsi-web/welcomePageAdditionalContent.html /usr/share/jitsi-meet/static/welcomePageAdditionalContent.html
COPY assets/meeting_portal_app/jitsi-web/meeting-portal-ui.css /usr/share/jitsi-meet/meeting-portal-ui.css
COPY assets/meeting_portal_app/jitsi-web/meeting-portal-ui.js /usr/share/jitsi-meet/meeting-portal-ui.js
COPY assets/meeting_portal_app/nginx-custom/meeting-portal.conf /usr/local/share/jitsi-custom/meeting-portal.conf
COPY docker/web/20-copy-meeting-portal-nginx-custom /etc/cont-init.d/20-copy-meeting-portal-nginx-custom

RUN chmod 755 /etc/cont-init.d/20-copy-meeting-portal-nginx-custom
