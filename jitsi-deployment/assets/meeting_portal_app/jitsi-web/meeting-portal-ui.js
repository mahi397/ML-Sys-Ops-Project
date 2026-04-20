(function () {
    const APP_PREFIX = "/meeting-portal";
    const SESSION_ENDPOINT = APP_PREFIX + "/api/session";
    const LOGIN_ENDPOINT = APP_PREFIX + "/api/login";
    const SIGNUP_ENDPOINT = APP_PREFIX + "/api/signup";
    const LOGOUT_ENDPOINT = APP_PREFIX + "/api/logout";
    const ROOM_AUTH_ENDPOINT = APP_PREFIX + "/api/room-auth-url";
    const ROOM_PRESENCE_ENDPOINT = APP_PREFIX + "/api/room-context/presence";
    const RECAPS_ENDPOINT = APP_PREFIX + "/recaps";   // recap change

    const ROOT_SHELL_ID = "meeting-portal-shell";
    const ROOT_STYLESHEET_ID = "meeting-portal-ui-stylesheet";
    const RECAPS_LINK_ID = "meeting-portal-recaps-link";   // recap change
    const UI_ASSET_VERSION = "20260420b"; // bump this value to force clients to reload the stylesheet
    const GUEST_MODE_STORAGE_KEY = "meeting-portal-guest-mode";
    const ROUTE_POLL_INTERVAL_MS = 500;


    let refreshScheduled = false;
    let refreshSequence = 0;
    let lastRouteKey = "";

    const state = {
        session: {
            authenticated: false,
            user_id: "",
            display_name: "",
            email: ""
        },
        guestMode: loadGuestMode(),
        overlayView: "loading",
        overlayError: "",
        statusMessage: "",
        pendingFormKind: ""
    };

    function loadGuestMode() {
        try {
            return window.sessionStorage.getItem(GUEST_MODE_STORAGE_KEY) === "1";
        } catch (error) {
            return false;
        }
    }

    function saveGuestMode(enabled) {
        try {
            if (enabled) {
                window.sessionStorage.setItem(GUEST_MODE_STORAGE_KEY, "1");
            } else {
                window.sessionStorage.removeItem(GUEST_MODE_STORAGE_KEY);
            }
        } catch (error) {
            return;
        }
    }

    function isRootPath() {
        return window.location.pathname === "/";
    }

    function currentRouteKey() {
        return window.location.pathname + window.location.search;
    }

    function isMeetingRoomPath() {
        return !isRootPath() && !window.location.pathname.startsWith(APP_PREFIX + "/");
    }

    function currentRoomPathName() {
        if (!isMeetingRoomPath()) {
            return "";
        }

        const rawPath = (window.location.pathname || "").replace(/^\/+/, "");
        const roomPathSegment = rawPath.split("/")[0] || "";
        if (!roomPathSegment) {
            return "";
        }

        try {
            return decodeURIComponent(roomPathSegment);
        } catch (error) {
            return roomPathSegment;
        }
    }

    function normalizeDisplayName(value) {
        return typeof value === "string" ? value.trim() : "";
    }

    function roomPresenceMarkerKey(roomName, jwtToken, displayName) {
        return "meeting-portal-room-presence:" + roomName + ":" + jwtToken.slice(-24) + ":" + normalizeDisplayName(displayName).toLowerCase();
    }

    function mergeCurrentSearchIntoRoomUrl(urlString) {
        const redirectUrl = new URL(urlString, window.location.origin);
        const searchParams = currentSearchParams();
        for (const [key, value] of searchParams.entries()) {
            if (key === "jwt") {
                continue;
            }
            redirectUrl.searchParams.append(key, value);
        }
        redirectUrl.hash = window.location.hash || "";
        return redirectUrl.toString();
    }

    function visibleElement(element) {
        return Boolean(element && element.offsetParent !== null);
    }

    function getShell() {
        return document.getElementById(ROOT_SHELL_ID);
    }

    function getStylesheet() {
        return document.getElementById(ROOT_STYLESHEET_ID);
    }

    function getRecapsLink() {
        return document.getElementById(RECAPS_LINK_ID);   // recap change
    }

    function ensureStylesheetMounted() {
        if (!isRootPath() || !document.head) {
            return null;
        }

        let stylesheet = getStylesheet();
        if (stylesheet instanceof HTMLLinkElement) {
            return stylesheet;
        }

        stylesheet = document.createElement("link");
        stylesheet.id = ROOT_STYLESHEET_ID;
        stylesheet.rel = "stylesheet";
        stylesheet.href = "/meeting-portal-ui.css?v=" + UI_ASSET_VERSION;
        document.head.appendChild(stylesheet);

        return stylesheet;
    }

    function removeStylesheet() {
        const stylesheet = getStylesheet();
        if (stylesheet) {
            stylesheet.remove();
        }
    }

    function removeShell() {
        const shell = getShell();
        if (shell) {
            shell.remove();
        }
    }

    function removeRecapsLink() {
        const link = getRecapsLink();    // recap change
        if (link) {
            link.remove();
        }
    }

    function ensureShellMounted() {
        if (!isRootPath()) {
            return null;
        }

        let shell = getShell();
        if (shell) {
            return shell;
        }

        const template = document.getElementById("welcome-page-additional-content-template");
        if (!(template instanceof HTMLTemplateElement) || !document.body) {
            return null;
        }

        const fragment = template.content.cloneNode(true);
        document.body.appendChild(fragment);
        return getShell();
    }
    
    function ensureRecapsLinkMounted() {             // recap change
        if (!isRootPath() || !document.body) {
            return null;
        }

        let link = getRecapsLink();
        if (!(link instanceof HTMLAnchorElement)) {
            link = document.createElement("a");
            link.id = RECAPS_LINK_ID;
            link.href = RECAPS_ENDPOINT;
            link.textContent = "Recaps";
            link.setAttribute("aria-label", "Open meeting recaps");
            document.body.appendChild(link);
        }

        link.hidden = !state.session.authenticated;
        return link;
    }



    function getRoomInput() {
        const candidates = Array.from(
            document.querySelectorAll('input[type="text"], input:not([type])')
        );

        return candidates.find((input) => {
            if (!visibleElement(input) || input.disabled || input.readOnly) {
                return false;
            }

            const placeholder = (input.getAttribute("placeholder") || "").toLowerCase();
            const ariaLabel = (input.getAttribute("aria-label") || "").toLowerCase();
            const name = (input.getAttribute("name") || "").toLowerCase();

            return (
                placeholder.includes("room") ||
                placeholder.includes("meeting") ||
                ariaLabel.includes("room") ||
                ariaLabel.includes("meeting") ||
                name.includes("room")
            ) || candidates.length === 1;
        }) || null;
    }

    function focusRoomInput() {
        const input = getRoomInput();
        if (input) {
            input.focus();
        }
    }

    function getRoomName() {
        const input = getRoomInput();
        return (input && input.value ? input.value : "").trim();
    }

    function currentSearchParams() {
        return new URLSearchParams(window.location.search || "");
    }

    function currentNextPath() {
        const nextPath = currentSearchParams().get("next");
        if (!nextPath || !nextPath.startsWith("/") || nextPath.startsWith("//")) {
            return "";
        }
        return nextPath;
    }

    function currentAuthMode() {
        const authMode = currentSearchParams().get("auth");
        return authMode === "login" || authMode === "signup" ? authMode : "";
    }

    function getJitsiState() {
        try {
            const store = window.APP && window.APP.store;
            if (!store || typeof store.getState !== "function") {
                return null;
            }

            return store.getState();
        } catch (error) {
            return null;
        }
    }

    function getCurrentLocalParticipant() {
        const state = getJitsiState();
        const participants = state && state["features/base/participants"];
        if (Array.isArray(participants)) {
            return participants.find((participant) => participant && (participant.local || participant.isLocal)) || null;
        }

        if (participants && typeof participants === "object") {
            const values = Object.values(participants);
            return values.find((participant) => participant && (participant.local || participant.isLocal)) || null;
        }

        return null;
    }

    function getCurrentMeetingDisplayName() {
        const localParticipant = getCurrentLocalParticipant();
        const participantDisplayName = normalizeDisplayName(
            (localParticipant && (localParticipant.name || localParticipant.displayName)) || ""
        );
        if (participantDisplayName) {
            return participantDisplayName;
        }

        const state = getJitsiState();
        const settings = state && state["features/base/settings"];
        return (
            normalizeDisplayName(settings && settings.displayName) ||
            normalizeDisplayName(settings && settings.userName) ||
            ""
        );
    }

    async function sendRoomPresenceFromJwt() {
        if (!isMeetingRoomPath()) {
            return;
        }

        const roomName = currentRoomPathName();
        const jwtToken = currentSearchParams().get("jwt") || "";
        if (!roomName || !jwtToken) {
            return;
        }

        const displayName = getCurrentMeetingDisplayName();
        const markerKey = roomPresenceMarkerKey(roomName, jwtToken, displayName);
        try {
            if (window.sessionStorage.getItem(markerKey) === "1") {
                return;
            }
        } catch (error) {
            // Ignore sessionStorage failures and still try to send presence.
        }

        try {
            const response = await fetch(ROOM_PRESENCE_ENDPOINT, {
                method: "POST",
                credentials: "same-origin",
                headers: {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    room_name: roomName,
                    jwt: jwtToken,
                    display_name: displayName
                }),
                keepalive: true
            });

            if (!response.ok) {
                return;
            }

            try {
                window.sessionStorage.setItem(markerKey, "1");
            } catch (error) {
                return;
            }
        } catch (error) {
            return;
        }
    }

    function hostLaunchUrl(roomName) {
        return roomName
            ? APP_PREFIX + "/host-launch/" + encodeURIComponent(roomName)
            : APP_PREFIX + "/host-launch";
    }

    function guestJoinUrl(roomName) {
        return APP_PREFIX + "/join/" + encodeURIComponent(roomName);
    }

    function loginUrl(roomName) {
        return "/?auth=login&next=" + encodeURIComponent(hostLaunchUrl(roomName));
    }

    function signupUrl(roomName) {
        return "/?auth=signup&next=" + encodeURIComponent(hostLaunchUrl(roomName));
    }

    async function fetchAuthenticatedRoomUrl(roomName) {
        const response = await fetch(
            ROOM_AUTH_ENDPOINT + "?room=" + encodeURIComponent(roomName),
            {
                credentials: "same-origin",
                cache: "no-store"
            }
        );
        if (!response.ok) {
            return null;
        }

        const payload = await parseJsonResponse(response);
        if (!payload.authenticated || !payload.room_url) {
            return null;
        }

        return mergeCurrentSearchIntoRoomUrl(payload.room_url);
    }

    async function navigateToAuthenticatedRoom(roomName) {
        const targetRoomName = roomName || getRoomName();

        try {
            const roomUrl = await fetchAuthenticatedRoomUrl(targetRoomName);
            if (roomUrl) {
                window.location.href = roomUrl;
                return;
            }
        } catch (error) {
            // Fall back to the existing host-launch redirect if the helper endpoint is unavailable.
        }

        window.location.href = hostLaunchUrl(targetRoomName);
    }

    async function parseJsonResponse(response) {
        try {
            return await response.json();
        } catch (error) {
            return {};
        }
    }

    async function fetchSession() {
        try {
            const response = await fetch(SESSION_ENDPOINT, {
                credentials: "same-origin",
                cache: "no-store"
            });

            if (!response.ok) {
                throw new Error("session request failed");
            }

            return await parseJsonResponse(response);
        } catch (error) {
            return {
                authenticated: false,
                login_url: loginUrl(""),
                signup_url: signupUrl(""),
                host_launch_url: hostLaunchUrl(""),
                guest_join_url: ""
            };
        }
    }

    async function postForm(endpoint, formData) {
        const body = new URLSearchParams();
        for (const [key, value] of formData.entries()) {
            if (typeof value === "string") {
                body.append(key, value);
            }
        }

        const response = await fetch(endpoint, {
            method: "POST",
            credentials: "same-origin",
            headers: {
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
            },
            body: body.toString()
        });

        const payload = await parseJsonResponse(response);
        return {
            ok: response.ok && payload.ok !== false,
            payload
        };
    }

    function setOverlayView(view, errorMessage) {
        state.overlayView = view;
        state.overlayError = errorMessage || "";
        renderAuthUI();
    }

    function closeOverlay() {
        state.overlayView = "closed";
        state.overlayError = "";
        renderAuthUI();
    }

    function showStatusMessage(message) {
        state.statusMessage = message;
        renderAuthUI();
    }

    function clearStatusMessage() {
        if (!state.statusMessage) {
            return;
        }
        state.statusMessage = "";
        renderAuthUI();
    }

    function resetForms() {
        const shell = getShell();
        if (!shell) {
            return;
        }

        const forms = shell.querySelectorAll("form[data-form-kind]");
        for (const form of forms) {
            form.reset();
        }
    }

    function updateOverlayCopy(titleElement, subtitleElement) {
        if (!titleElement || !subtitleElement) {
            return;
        }

        if (state.overlayView === "loading") {
            titleElement.textContent = "Checking your Jitsi session";
            subtitleElement.textContent = "Loading account access options.";
            return;
        }

        if (state.overlayView === "login") {
            titleElement.textContent = "Log in to create a meeting";
            subtitleElement.textContent = "Use your existing account. Once authenticated, the normal Jitsi welcome page stays available underneath.";
            return;
        }

        if (state.overlayView === "signup") {
            titleElement.textContent = "Create an account";
            subtitleElement.textContent = "New accounts are stored in the shared users table and can immediately host authenticated meetings.";
            return;
        }

        titleElement.textContent = "Choose how to continue";
        subtitleElement.textContent = "Authenticated users can create meetings. Guests can still join an existing room.";
    }

    function renderStatusChip(shell) {
        const status = shell.querySelector("#meeting-portal-status");
        const title = status ? status.querySelector(".meeting-portal-status-title") : null;
        const subtitle = status ? status.querySelector(".meeting-portal-status-subtitle") : null;
        const message = status ? status.querySelector(".meeting-portal-status-message") : null;
        const loginButton = status ? status.querySelector('[data-action="open-login"]') : null;
        const logoutButton = status ? status.querySelector('[data-action="logout"]') : null;

        if (!status || !title || !subtitle || !message || !loginButton || !logoutButton) {
            return;
        }

        const shouldShow = state.session.authenticated || state.guestMode;
        status.hidden = !shouldShow;
        if (!shouldShow) {
            return;
        }

        if (state.session.authenticated) {
            title.textContent = "Logged in as " + (state.session.display_name || state.session.user_id || "user");
            subtitle.textContent = "Authenticated host controls are enabled on this Jitsi welcome page.";
            loginButton.hidden = true;
            logoutButton.hidden = false;
        } else {
            title.textContent = "Guest mode";
            subtitle.textContent = "Enter an existing room name before you continue as a guest.";
            loginButton.hidden = false;
            logoutButton.hidden = true;
        }

        message.hidden = !state.statusMessage;
        message.textContent = state.statusMessage;
    }

    function renderPanels(shell) {
        const gate = shell.querySelector("#meeting-portal-gate");
        const title = shell.querySelector("#meeting-portal-title");
        const subtitle = shell.querySelector("#meeting-portal-subtitle");
        const error = shell.querySelector("#meeting-portal-error");
        const panels = shell.querySelectorAll("[data-panel]");
        const shouldShowGate = state.overlayView !== "closed";

        gate.hidden = !shouldShowGate;
        if (!shouldShowGate) {
            return;
        }

        updateOverlayCopy(title, subtitle);

        error.hidden = !state.overlayError;
        error.textContent = state.overlayError;

        for (const panel of panels) {
            panel.hidden = panel.getAttribute("data-panel") !== state.overlayView;
        }

        const forms = shell.querySelectorAll("form[data-form-kind]");
        for (const form of forms) {
            const formKind = form.getAttribute("data-form-kind") || "";
            const isPending = state.pendingFormKind === formKind;
            const inputs = form.querySelectorAll("input, button");
            for (const input of inputs) {
                input.disabled = isPending;
            }
        }
    }

    function renderAuthUI() {
        if (!isRootPath()) {
            removeShell();
            removeStylesheet();
            removeRecapsLink();    // recap change
            return;
        }

        ensureStylesheetMounted();

        const shell = ensureShellMounted();
        if (!shell) {
            return;
        }

        shell.hidden = false;
        shell.dataset.view = state.overlayView;
        renderPanels(shell);
        renderStatusChip(shell);
        ensureRecapsLinkMounted();   // recap change
    }

    async function refreshSessionState() {
        const sequence = ++refreshSequence;
        if (!isRootPath()) {
            return;
        }

        const session = await fetchSession();
        if (sequence !== refreshSequence) {
            return;
        }

        state.session = session;
        state.pendingFormKind = "";

        if (session.authenticated) {
            const nextPath = currentNextPath();
            if (nextPath && nextPath !== window.location.pathname) {
                window.location.href = nextPath;
                return;
            }
            state.guestMode = false;
            saveGuestMode(false);
            state.statusMessage = "";
            closeOverlay();
            return;
        }

        const authMode = currentAuthMode();
        if (authMode) {
            state.guestMode = false;
            saveGuestMode(false);
            state.statusMessage = "";
            setOverlayView(authMode, "");
            return;
        }

        if (state.guestMode) {
            state.statusMessage = "";
            closeOverlay();
            return;
        }

        setOverlayView("choice", "");
    }

    function scheduleSessionRefresh() {
        if (refreshScheduled) {
            return;
        }

        refreshScheduled = true;
        window.setTimeout(() => {
            refreshScheduled = false;
            void refreshSessionState();
        }, 60);
    }

    async function handleFormSubmission(form) {
        const formKind = form.getAttribute("data-form-kind");
        if (formKind !== "login" && formKind !== "signup") {
            return;
        }

        const endpoint = formKind === "login" ? LOGIN_ENDPOINT : SIGNUP_ENDPOINT;
        const formData = new FormData(form);
        formData.set("room", getRoomName());
        state.pendingFormKind = formKind;
        state.overlayError = "";
        renderAuthUI();

        const result = await postForm(endpoint, formData);
        state.pendingFormKind = "";

        if (!result.ok) {
            setOverlayView(formKind, result.payload.error || "We could not complete that request.");
            return;
        }

        state.session = result.payload;
        state.guestMode = false;
        saveGuestMode(false);
        state.statusMessage = "";
        resetForms();
        const nextPath = currentNextPath();
        if (nextPath && nextPath !== window.location.pathname) {
            window.location.href = nextPath;
            return;
        }
        closeOverlay();
    }

    async function handleLogout() {
        const formData = new FormData();
        formData.set("room", getRoomName());
        await postForm(LOGOUT_ENDPOINT, formData);

        state.session = {
            authenticated: false,
            user_id: "",
            display_name: "",
            email: ""
        };
        state.guestMode = false;
        saveGuestMode(false);
        resetForms();
        setOverlayView("choice", "");
    }

    function matchesActionLabel(element, phrases) {
        const label = (element.textContent || "").trim().toLowerCase();
        if (!label) {
            return false;
        }

        return phrases.some((phrase) => label.includes(phrase));
    }

    function isGuestAllowed() {
        return state.guestMode && !state.session.authenticated;
    }

    function handleStartMeetingClick(event) {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }

        const element = target.closest("button, a");
        if (!element || !visibleElement(element)) {
            return;
        }

        if (!matchesActionLabel(element, ["start meeting"])) {
            return;
        }

        if (!isRootPath()) {
            return;
        }

        const roomName = getRoomName();

        if (state.session.authenticated) {
            event.preventDefault();
            event.stopPropagation();
            void navigateToAuthenticatedRoom(roomName);
            return;
        }

        if (isGuestAllowed()) {
            event.preventDefault();
            event.stopPropagation();

            if (!roomName) {
                showStatusMessage("Guests must enter an existing room name before continuing.");
                focusRoomInput();
                return;
            }

            clearStatusMessage();
            window.location.href = guestJoinUrl(roomName);
            return;
        }
    }

    function handleActionClick(event) {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }

        const actionElement = target.closest("[data-action]");
        const shell = getShell();
        if (!actionElement || !shell || !shell.contains(actionElement) || !visibleElement(actionElement)) {
            return;
        }

        const action = actionElement.getAttribute("data-action");
        if (!action) {
            return;
        }

        if (action === "open-login") {
            event.preventDefault();
            state.guestMode = false;
            saveGuestMode(false);
            setOverlayView("login", "");
            return;
        }

        if (action === "open-signup") {
            event.preventDefault();
            state.guestMode = false;
            saveGuestMode(false);
            setOverlayView("signup", "");
            return;
        }

        if (action === "back-to-choice") {
            event.preventDefault();
            setOverlayView("choice", "");
            return;
        }

        if (action === "continue-guest") {
            event.preventDefault();
            state.guestMode = true;
            saveGuestMode(true);
            state.statusMessage = "";
            closeOverlay();
            return;
        }

        if (action === "logout") {
            event.preventDefault();
            void handleLogout();
        }
    }

    function handleSubmit(event) {
        const target = event.target;
        if (!(target instanceof HTMLFormElement)) {
            return;
        }

        const shell = getShell();
        if (!shell || !shell.contains(target) || !target.matches("form[data-form-kind]")) {
            return;
        }

        event.preventDefault();
        void handleFormSubmission(target);
    }

    function tick() {
        const routeKey = currentRouteKey();
        if (routeKey !== lastRouteKey) {
            lastRouteKey = routeKey;
            if (isRootPath()) {
                ensureShellMounted();
                scheduleSessionRefresh();
            } else {
                renderAuthUI();
            }
        }

        if (isMeetingRoomPath()) {
            void sendRoomPresenceFromJwt();
        }
    }

    function boot() {
        lastRouteKey = currentRouteKey();
        document.addEventListener("click", handleActionClick, true);
        document.addEventListener("click", handleStartMeetingClick, true);
        document.addEventListener("submit", handleSubmit, true);
        window.addEventListener("focus", scheduleSessionRefresh);
        window.addEventListener("pageshow", scheduleSessionRefresh);
        document.addEventListener("visibilitychange", () => {
            if (!document.hidden) {
                scheduleSessionRefresh();
            }
        });

        window.setInterval(tick, ROUTE_POLL_INTERVAL_MS);
        renderAuthUI();
        if (isRootPath()) {
            scheduleSessionRefresh();
        } else {
            void sendRoomPresenceFromJwt();
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot, { once: true });
    } else {
        boot();
    }
})();
