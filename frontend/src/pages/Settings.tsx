import { useEffect, useState } from "react";
import { api, type AutonomySettings } from "../lib/api";

const TOGGLES: { key: keyof AutonomySettings; label: string; hint: string }[] = [
  { key: "auto_search", label: "Auto search", hint: "Discover jobs from enabled searches." },
  { key: "auto_match", label: "Auto match", hint: "Score each posting against your profile." },
  {
    key: "auto_optimize_resume",
    label: "Auto optimize resume",
    hint: "Generate a tailored CV per job. Never adds facts you do not have.",
  },
  { key: "auto_apply", label: "Auto apply", hint: "Open and fill applications." },
  {
    key: "auto_submit",
    label: "Auto submit",
    hint: "Send the application once pre-submission validation passes.",
  },
  {
    key: "review_before_submit",
    label: "Review before submit",
    hint: "Ask you to approve each application before it is sent.",
  },
  {
    key: "delete_temp_resumes",
    label: "Delete temporary CVs",
    hint: "Remove the generated file after submission is confirmed. The recipe is kept.",
  },
];

export function SettingsPage() {
  const [settings, setSettings] = useState<AutonomySettings | null>(null);
  const [storage, setStorage] = useState<Record<string, { count: number; bytes: number }>>({});
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getAutonomy().then(setSettings).catch((e) => setError(String(e.message ?? e)));
    api.storage().then((r) => setStorage(r.usage)).catch(() => undefined);
    api.health().then(setHealth).catch(() => undefined);
  }, []);

  async function save(next: AutonomySettings) {
    setSettings(next);
    try {
      await api.setAutonomy(next);
      setStatus("Saved.");
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function collect(policy: string) {
    const result = await api.collectStorage(policy);
    setStatus(
      `Deleted ${result.deleted} file(s), freed ${(result.freed_bytes / 1024).toFixed(0)} KB.`,
    );
    setStorage((await api.storage()).usage);
  }

  const roles =
    (health?.ai as { roles?: Record<string, { model: string; provider: string; available: boolean }> })
      ?.roles ?? {};
  const resources = (health?.ai as { resources?: { used_mb: number; budget_mb: number } })
    ?.resources;

  if (!settings) return <div className="card empty">Loading settings…</div>;

  return (
    <>
      <h1>Settings</h1>
      <p className="page-sub">
        Autonomy is a dial, not a switch. Everything runs locally unless you configure a
        remote model endpoint.
      </p>

      {error && <div className="banner danger">{error}</div>}
      {status && <div className="banner info">{status}</div>}

      <div className="card">
        <h2>Safety</h2>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.dry_run}
            onChange={(e) => void save({ ...settings, dry_run: e.target.checked })}
          />
          <span>
            <strong>Dry run</strong> — do everything except actually submit
          </span>
        </label>
        <p className="muted" style={{ marginTop: 4 }}>
          With dry run on, the agent still searches, matches, generates CVs and fills forms.
          Only the final submit is blocked. Recommended until you trust the results.
        </p>
      </div>

      <div className="card">
        <h2>Autonomy</h2>
        {TOGGLES.map((toggle) => (
          <div key={toggle.key} style={{ marginBottom: 10 }}>
            <label className="check">
              <input
                type="checkbox"
                checked={Boolean(settings[toggle.key])}
                onChange={(e) => void save({ ...settings, [toggle.key]: e.target.checked })}
              />
              <span>{toggle.label}</span>
            </label>
            <div className="muted" style={{ marginLeft: 26, fontSize: 12 }}>
              {toggle.hint}
            </div>
          </div>
        ))}

        <div className="grid cols-3" style={{ marginTop: 16 }}>
          <label className="field">
            <span>Minimum match ({Math.round(settings.minimum_match_score * 100)}%)</span>
            <input
              type="number"
              min={0}
              max={100}
              value={Math.round(settings.minimum_match_score * 100)}
              onChange={(e) =>
                void save({ ...settings, minimum_match_score: Number(e.target.value) / 100 })
              }
            />
          </label>
          <label className="field">
            <span>Daily application limit</span>
            <input
              type="number"
              min={0}
              value={settings.daily_application_limit}
              onChange={(e) =>
                void save({ ...settings, daily_application_limit: Number(e.target.value) })
              }
            />
          </label>
          <label className="field">
            <span>Session application limit</span>
            <input
              type="number"
              min={0}
              value={settings.session_application_limit}
              onChange={(e) =>
                void save({ ...settings, session_application_limit: Number(e.target.value) })
              }
            />
          </label>
        </div>
      </div>

      <div className="card">
        <h2>External accounts</h2>
        <label className="field">
          <span>When an application requires an account CVnator does not have</span>
          <select
            value={settings.account_policy}
            onChange={(e) =>
              void save({
                ...settings,
                account_policy: e.target.value as AutonomySettings["account_policy"],
              })
            }
          >
            <option value="never">Never create one — abandon the application</option>
            <option value="ask">Ask me every time</option>
            <option value="auto">Create automatically with my approved credential policy</option>
          </select>
          <small>
            Passwords are stored in your operating system's credential vault and are never
            written to the database, the logs, or a model prompt. Additional verification
            (CAPTCHA, MFA, email confirmation) will still require you.
          </small>
        </label>
      </div>

      <div className="card">
        <h2>Local models</h2>
        {Object.entries(roles).map(([role, info]) => (
          <div className="spread" key={role} style={{ marginBottom: 6 }}>
            <span>
              <strong>{role}</strong>{" "}
              <span className="muted mono">
                {info.provider}/{info.model}
              </span>
            </span>
            <span className={`badge ${info.available ? "ok" : "danger"}`}>
              {info.available ? "available" : "unavailable"}
            </span>
          </div>
        ))}
        {resources && (
          <p className="muted" style={{ marginBottom: 0 }}>
            VRAM in use: {resources.used_mb} MB of {resources.budget_mb} MB budget. Models are
            loaded and evicted on demand so they never compete for the card.
          </p>
        )}
      </div>

      <div className="card">
        <h2>Storage</h2>
        {Object.keys(storage).length === 0 ? (
          <p className="muted">No tracked artifacts.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Kind</th>
                <th>Files</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(storage).map(([kind, usage]) => (
                <tr key={kind}>
                  <td>{kind}</td>
                  <td>{usage.count}</td>
                  <td>{(usage.bytes / 1024).toFixed(0)} KB</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="row" style={{ marginTop: 12 }}>
          <button className="secondary" onClick={() => void collect("delete_after_24h")}>
            Clean up files older than 24h
          </button>
          <button className="secondary" onClick={() => void collect("delete_after_7d")}>
            Older than 7 days
          </button>
        </div>
        <p className="muted" style={{ marginBottom: 0 }}>
          Your master CV, anything marked permanent, and files currently in use are never
          deleted.
        </p>
      </div>
    </>
  );
}
