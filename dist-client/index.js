// dsh-video-retrieval client half — the mode's dedicated three-column console
// (P5). Hand-maintained classic script in the exact `window.__ModuleLoader__`
// bundle format shipped client packages use (see dsh-client-ui-layout/lib/client.js).
// Plain JS + React.createElement only; platform modules come from `require`.
//
// Surface: the fusion three-column console rendered INSIDE the app frame's
// main area (right of the sidebar — the sidebar stays visible and clickable).
// Entry points: a 视频检索 button in the sidebar footer, and auto-open when
// the current session runs on `video-retrieval` and is blank.
//
// Open/close signaling goes through a window CustomEvent (`vr-console`) so the
// state is shared across bundle instances and debuggable from DevTools.
window.__ModuleLoader__.load({
	id: "dsh-video-retrieval",
	factory: (require) => {
		var module = { exports: {} };
		var exports = module.exports;
		Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
		let react = require("react");

		const CONSOLE_URL = "http://127.0.0.1:8788/";
		const PRESET_ID = "video-retrieval";
		const EVT = "vr-console";

		if (window.__vrConsole === void 0) window.__vrConsole = { lastClosedSessionId: void 0 };
		function setOpen(value) {
			window.dispatchEvent(new CustomEvent(EVT, { detail: { open: value } }));
		}

		const barStyle = {
			display: "flex",
			alignItems: "center",
			gap: "8px",
			padding: "8px 12px",
			borderBottom: "1px solid var(--dsw-alias-border-l2, #e3e5ea)",
			background: "var(--dsw-alias-bg-layer-3, #ffffff)"
		};
		const titleStyle = {
			fontSize: "13px",
			fontWeight: 600,
			color: "var(--dsw-alias-label-primary, #1b1d22)",
			flex: 1
		};
		const btnStyle = {
			appearance: "none",
			border: "1px solid var(--dsw-alias-border-l2, #e3e5ea)",
			background: "var(--dsw-alias-bg-layer-1, #ffffff)",
			color: "var(--dsw-alias-label-secondary, #4a4f59)",
			borderRadius: "7px",
			padding: "4px 10px",
			fontSize: "12px",
			cursor: "pointer"
		};

		function ConsoleOverlay(props) {
			const [isOpen, setOpenState] = react.useState(false);
			const currentId = props.useSessions ? props.useSessions(function (s) { return s.current; }) : void 0;
			const blank = props.useSessions ? props.useSessions(function (s) {
				const cur = s.current;
				return cur !== void 0 && s.byId[cur] !== void 0 ? s.byId[cur].blank : void 0;
			}) : void 0;
			const preset = props.useSessions ? props.useSessions(function (s) {
				const cur = s.current;
				return cur !== void 0 && s.byId[cur] !== void 0 ? s.byId[cur].agentPreset : void 0;
			}) : void 0;

			// Listen for the shared open/close signal.
			react.useEffect(function () {
				const handler = function (event) {
					setOpenState(event.detail && event.detail.open === true);
				};
				window.addEventListener(EVT, handler);
				return function () { window.removeEventListener(EVT, handler); };
			}, []);

			// Auto-open: first arrival on a blank video-retrieval session.
			react.useEffect(function () {
				if (isOpen) return;
				if (preset !== PRESET_ID || blank !== true) return;
				if (currentId === window.__vrConsole.lastClosedSessionId) return;
				setOpenState(true);
			}, [isOpen, currentId, preset, blank]);

			// Switching conversation from the sidebar closes the console and
			// returns to the DeepSeek chat (the sidebar must stay usable).
			const prevIdRef = react.useRef(currentId);
			react.useEffect(function () {
				if (prevIdRef.current !== void 0 && prevIdRef.current !== currentId && isOpen) {
					window.__vrConsole.lastClosedSessionId = currentId;
					setOpen(false);
				}
				prevIdRef.current = currentId;
			}, [currentId, isOpen]);

			// Measure the left sidebar width so the console fills the MAIN area
			// (right of the sidebar) instead of covering the whole frame.
			// Live DOM: panel → slot-list wrapper → overlayLayer → frameRoot;
			// frameRoot's first element child is the sidebar column.
			const panelRef = react.useRef(null);
			const [leftOffset, setLeftOffset] = react.useState(0);
			react.useEffect(function () {
				if (!isOpen) return;
				const wrapper = panelRef.current && panelRef.current.parentElement;
				const overlayEl = wrapper && wrapper.parentElement;
				const frame = overlayEl && overlayEl.parentElement;
				if (!frame) return;
				const update = function () {
					const first = frame.firstElementChild;
					setLeftOffset(first && first !== overlayEl ? first.getBoundingClientRect().width : 0);
				};
				update();
				const ro = new ResizeObserver(update);
				ro.observe(frame);
				for (let i = 0; i < frame.children.length; i++) ro.observe(frame.children[i]);
				return function () { ro.disconnect(); };
			}, [isOpen]);

			if (!isOpen) return null;

			return react.createElement("div", {
				ref: panelRef,
				style: {
					position: "absolute",
					top: 0,
					bottom: 0,
					left: leftOffset,
					right: 0,
					display: "flex",
					flexDirection: "column",
					background: "var(--dsw-alias-bg-layer-2, #f5f6f8)",
					boxShadow: "0 0 40px rgba(0,0,0,.35)",
					borderLeft: "1px solid var(--dsw-alias-border-l2, #e3e5ea)"
				}
			},
				react.createElement("div", { style: barStyle },
					react.createElement("span", { style: titleStyle },
						"视频检索控制台 · 云边协同 · 视频不出本机"),
					react.createElement("a", {
						href: CONSOLE_URL,
						target: "_blank",
						rel: "noreferrer",
						style: btnStyle,
						title: "在新标签页打开"
					}, "新标签页打开"),
					react.createElement("button", {
						style: btnStyle,
						title: "关闭控制台，回到对话界面（可从左侧边栏重新打开）",
						onClick: function () {
							window.__vrConsole.lastClosedSessionId = currentId;
							setOpen(false);
						}
					}, "关闭")
				),
				react.createElement("iframe", {
					src: CONSOLE_URL,
					title: "视频检索控制台",
					style: { flex: 1, border: 0, width: "100%", height: "100%", background: "#ffffff" }
				})
			);
		}

		function SidebarOpenAction() {
			return react.createElement("button", {
				style: btnStyle,
				title: "打开视频检索控制台（嵌入主区域，保留左侧边栏）",
				onClick: function () {
					window.__vrConsole.lastClosedSessionId = void 0;
					setOpen(true);
				}
			}, "视频检索");
		}

		function apply(ctx) {
			const slots = ctx.get("slots");
			if (slots === void 0) return;
			slots.inject("shell.overlay", function () {
				return slots.register({ name: "shell.overlay", id: "vr-console" }, ConsoleOverlay);
			});
			slots.inject("sidebar.footer.action", function () {
				return slots.register({ name: "sidebar.footer.action", id: "vr-console-open" }, SidebarOpenAction);
			});
		}

		exports.apply = apply;
		return module.exports;
	}
});
