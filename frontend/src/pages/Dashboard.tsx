import { useEffect, useState } from "react";
import { api, type LogEntry, type SessionSnapshot, type UserAction } from "../lib/api";

type Control = (action: () => Promise<SessionSnapshot>) => Promise<void>;

const METRIC_LABELS: [keyof NonNullable<SessionSnapshot["metrics"]>, string][] = [
  ["jobs_discovered", "Jobs found"],
  ["jobs_analyzed", "Analyzed"],
  ["jobs_matched", "Matches"],
  ["jobs_rejected", "Rejected"],
  ["applications_started", "Applications"],
  ["applications_completed", "Submitted"],
  ["applications_failed", "Failures"],
  ["applications_waiting_user", "Waiting on you"],
  ["resumes_generated", "CVs generated"],
];

export function Dashboard({
  snapshot,
  actions,
  logs,
  error,
  control,
}: {
  snapshot: SessionSnapshot | null;
  actions: UserAction[];
  logs: LogEntry[];
  error: string | null;
  control: Control;
}) {
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const status = snapshot?.status ?? "stopped";
  const running = status === "running";
  const paused = status === "paused" || status === "user_control";
  const metrics = snapshot?.metrics ?? {};
  const activity = snapshot?.activity ?? {};
  const pending = actions.filter((a) => !a.resolved);

  const aiRoles =
    (health?.ai as { roles?: Record<string, { model: string; available: boolean }> } | undefined)
      ?.roles ?? {};
  const aiDown = Object.values(aiRoles).filter((r) => !r.available);

  return (
    <>
      <h1>Dashboard</h1>
      <p className="page-sub">
        Configure once, then let the agent work. It will ask only when it genuinely
        cannot proceed on its own.
      </p>

      {error && <div className="banner danger">{error}</div>}
      {aiDown.length > 0 && (
        <div className="banner warn">
          {aiDown.length} model role(s) unavailable. Check that Ollama is running and the
          configured models are pulled.
        </div>
      )}
      {snapshot?.autonomy?.dry_run && (
        <div className="banner info">
          <strong>Dry run is on.</strong> The agent will search, match, tailor CVs and fill
          forms, but will not submit anything.
        </div>
      )}

      <div className="card">
        <div className="spread">
          <div>
            <h2 style={{ marginBottom: 2 }}>Session</h2>
            <span className="muted">
              {running
                ? "Running autonomously"
                : paused
                  ? "Paused — the browser stays open"
                  : "Stopped"}
            </span>
          </div>
          <div className="row">
            {!running && !paused && (
              <button onClick={() => control(api.startSession)}>Start</button>
            )}
            {running && (
              <button className="secondary" onClick={() => control(api.pauseSession)}>
                Pause
              </button>
            )}
            {paused && <button onClick={() => control(api.resumeSession)}>Resume</button>}
            {(running || paused) && (
              <>
                <button className="secondary" onClick={() => control(api.skipJob)}>
                  Skip job
                </button>
                {snapshot?.browser_user_control ? (
                  <button className="secondary" onClick={() => control(api.returnControl)}>
                    Return control
                  </button>
                ) : (
                  <button className="secondary" onClick={() => control(api.takeControl)}>
                    Take control
                  </button>
                )}
                <button className="danger" onClick={() => control(api.stopSession)}>
                  Stop
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      {activity.stage && activity.stage !== "idle" && (
        <div className="card">
          <h2>Current activity</h2>
          <div className="spread">
            <div>
              <div style={{ fontSize: 17, fontWeight: 650 }}>
                {activity.job_title || "—"}
              </div>
              <div className="muted">
                {activity.company || "—"}
                {activity.platform ? ` · ${activity.platform}` : ""}
              </div>
            </div>
            <span className="badge accent">{activity.stage}</span>
          </div>
          <div className="grid cols-2" style={{ marginTop: 14 }}>
            <Meter label="Job match" percent={activity.match_percent ?? null} />
            <Meter label="ATS estimate" percent={activity.ats_percent ?? null} />
          </div>
          {activity.step && (
            <p className="muted" style={{ marginTop: 12, marginBottom: 0 }}>
              {activity.step}
            </p>
          )}
        </div>
      )}

      {pending.length > 0 && (
        <div className="card" style={{ borderColor: "var(--warn)" }}>
          <h2>Action required ({pending.length})</h2>
          {pending.slice(0, 3).map((action) => (
            <div key={action.id} className="spread" style={{ marginBottom: 8 }}>
              <div>
                <strong>{action.kind.replace(/_/g, " ").toLowerCase()}</strong>
                <div className="muted">
                  {action.company || action.platform} — {action.problem}
                </div>
              </div>
              <a className="badge warn" href="/actions">
                Resolve
              </a>
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <h2>Statistics</h2>
        <div className="grid cols-4">
          {METRIC_LABELS.map(([key, label]) => (
            <div className="stat" key={key}>
              <div className="value">{(metrics as Record<string, number>)[key] ?? 0}</div>
              <div className="label">{label}</div>
            </div>
          ))}
        </div>
        {snapshot?.queues && Object.keys(snapshot.queues).length > 0 && (
          <p className="muted" style={{ marginTop: 12, marginBottom: 0 }}>
            Queues —{" "}
            {Object.entries(snapshot.queues)
              .map(([name, size]) => `${name}: ${size}`)
              .join(" · ")}
          </p>
        )}
      </div>

      <div className="card">
        <h2>Agent activity log</h2>
        <div className="logs">
          {logs.length === 0 && <div className="muted">No activity yet.</div>}
          {logs.slice(-150).map((entry, index) => (
            <div key={index} className={`lvl-${entry.level}`}>
              {entry.message}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

function Meter({ label, percent }: { label: string; percent: number | null }) {
  return (
    <div>
      <div className="spread" style={{ marginBottom: 4 }}>
        <span className="muted" style={{ fontSize: 12 }}>
          {label}
        </span>
        <strong>{percent === null ? "—" : `${percent}%`}</strong>
      </div>
      <div className="meter">
        <span style={{ width: `${percent ?? 0}%` }} />
      </div>
    </div>
  );
}
