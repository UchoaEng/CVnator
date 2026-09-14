import { useState } from "react";
import { api, type UserAction } from "../lib/api";

/** Interventions the agent genuinely cannot do itself — never busywork. */
const GUIDANCE: Record<string, string> = {
  CAPTCHA:
    "CVnator does not solve or bypass CAPTCHAs. Complete it in the open browser window, then mark this resolved.",
  MFA: "Complete the two-factor prompt in the browser, then mark this resolved.",
  SMS_CODE: "Enter the SMS code in the browser, then mark this resolved.",
  EMAIL_CONFIRMATION: "Confirm the email, then mark this resolved.",
  PASSKEY: "Approve the passkey prompt on your device, then mark this resolved.",
  LOGIN_REQUIRED:
    "Sign in yourself in the open browser. The session is remembered for next time.",
  ACCOUNT_CREATION_APPROVAL:
    "The agent wants to create an account on this platform. Approve or decline.",
  LEGAL_CONSENT: "Accepting terms is your decision, not the agent's. Approve or decline.",
  SUBMIT_APPROVAL: "Review the filled application in the browser, then approve or decline.",
  UNKNOWN_PERSONAL_INFORMATION:
    "The agent needs a fact it cannot derive from your profile. Your answer is remembered for next time.",
  NAVIGATION_FAILURE:
    "The agent is stuck. Take control of the browser to move it forward, or decline to abandon this application.",
  PLATFORM_BLOCKED:
    "The platform signalled that it is blocking automation. CVnator will not try to work around this.",
};

export function ActionsPage({
  actions,
  onResolved,
}: {
  actions: UserAction[];
  onResolved: () => void;
}) {
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pending = actions.filter((a) => !a.resolved);

  async function respond(action: UserAction, answer: string | null) {
    setBusy(action.id);
    try {
      const result = await api.answerUserAction(action.id, answer);
      if (!result.delivered) {
        setError(
          "The answer was recorded, but no running session was waiting for it. " +
            "The application may have moved on or the session may have stopped.",
        );
      } else {
        setError(null);
      }
      onResolved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <h1>Action Required</h1>
      <p className="page-sub">
        The agent handles what it can and stops here for what it cannot. Answers you give
        are remembered, so the same question is not asked twice.
      </p>

      {error && <div className="banner warn">{error}</div>}

      {pending.length === 0 ? (
        <div className="card empty">Nothing needs you right now.</div>
      ) : (
        pending.map((action) => (
          <div className="card" key={action.id} style={{ borderColor: "var(--warn)" }}>
            <div className="spread">
              <div>
                <span className="badge warn">{action.kind.replace(/_/g, " ").toLowerCase()}</span>
                <h2 style={{ margin: "8px 0 2px" }}>
                  {action.job_title || "Session"}
                  {action.company ? ` — ${action.company}` : ""}
                </h2>
                <div className="muted">{action.platform}</div>
              </div>
              <span className="muted" style={{ fontSize: 12 }}>
                {new Date(action.created_at).toLocaleTimeString()}
              </span>
            </div>

            <p style={{ marginBottom: 6 }}>
              <strong>Problem:</strong> {action.problem}
            </p>
            <p style={{ marginTop: 0 }}>
              <strong>Needed:</strong> {action.action_needed}
            </p>
            {GUIDANCE[action.kind] && (
              <div className="banner info">{GUIDANCE[action.kind]}</div>
            )}

            {action.question && (
              <label className="field">
                <span>{action.question}</span>
                <input
                  type="text"
                  value={answers[action.id] ?? ""}
                  placeholder="Your answer…"
                  onChange={(e) =>
                    setAnswers({ ...answers, [action.id]: e.target.value })
                  }
                />
              </label>
            )}

            <div className="row">
              <button
                disabled={busy === action.id}
                onClick={() =>
                  void respond(action, action.question ? (answers[action.id] ?? "") : "done")
                }
              >
                {action.question ? "Submit answer" : "I've done it — continue"}
              </button>
              <button
                className="secondary"
                disabled={busy === action.id}
                onClick={() => void respond(action, "no")}
              >
                Decline / skip this job
              </button>
            </div>
          </div>
        ))
      )}
    </>
  );
}
