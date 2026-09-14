import { NavLink, Outlet } from "react-router-dom";
import type { SessionSnapshot, UserAction } from "../lib/api";

const STATUS_TONE: Record<string, string> = {
  running: "ok",
  paused: "warn",
  user_control: "warn",
  stopping: "warn",
  stopped: "",
};

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/profile", label: "Profile" },
  { to: "/searches", label: "Job Searches" },
  { to: "/applications", label: "Applications" },
  { to: "/actions", label: "Action Required" },
  { to: "/settings", label: "Settings" },
];

export function Shell({
  snapshot,
  actions,
  connected,
}: {
  snapshot: SessionSnapshot | null;
  actions: UserAction[];
  connected: boolean;
}) {
  const status = snapshot?.status ?? "stopped";
  const pending = actions.filter((a) => !a.resolved).length;

  return (
    <div className="layout">
      <nav className="sidebar">
        <div className="brand">
          CVnator
          <small>autonomous application agent</small>
        </div>
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
          >
            <span>{item.label}</span>
            {item.to === "/actions" && pending > 0 && (
              <span className="badge warn">{pending}</span>
            )}
          </NavLink>
        ))}
        <div style={{ marginTop: "auto", padding: "12px 10px 0" }}>
          <div className={`badge ${STATUS_TONE[status] ?? ""}`}>{status.replace("_", " ")}</div>
          {snapshot?.autonomy?.dry_run && (
            <div className="badge accent" style={{ marginTop: 6 }}>
              dry run
            </div>
          )}
          <div className="muted" style={{ fontSize: 11, marginTop: 8 }}>
            {connected ? "live updates on" : "live updates reconnecting…"}
          </div>
        </div>
      </nav>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
