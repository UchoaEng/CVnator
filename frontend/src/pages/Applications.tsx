import { Fragment, useEffect, useMemo, useState } from "react";
import { api, type ApplicationSummary } from "../lib/api";

const STATE_TONE: Record<string, string> = {
  SUBMITTED: "ok",
  REVIEW_PENDING: "warn",
  WAITING_USER: "warn",
  FAILED: "danger",
  ABANDONED: "danger",
  REJECTED: "",
};

export function ApplicationsPage() {
  const [applications, setApplications] = useState<ApplicationSummary[]>([]);
  const [filter, setFilter] = useState("");
  const [expanded, setExpanded] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () =>
      api
        .listApplications(200)
        .then(setApplications)
        .catch((e) => setError(String(e.message ?? e)));
    void load();
    const timer = window.setInterval(load, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return applications;
    return applications.filter((a) =>
      `${a.job.role} ${a.job.company} ${a.state} ${a.platform}`.toLowerCase().includes(needle),
    );
  }, [applications, filter]);

  return (
    <>
      <h1>Applications</h1>
      <p className="page-sub">
        Every attempt is recorded, including the ones that failed. An application is marked
        submitted only when the page actually showed evidence of it.
      </p>

      {error && <div className="banner danger">{error}</div>}

      <div className="card">
        <input
          type="text"
          placeholder="Filter by role, company, state or platform…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>

      {visible.length === 0 ? (
        <div className="card empty">No applications yet.</div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <table>
            <thead>
              <tr>
                <th>Role</th>
                <th>Company</th>
                <th>Match</th>
                <th>State</th>
                <th>Started</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {visible.map((application) => (
                <Fragment key={application.id}>
                  <tr>
                    <td>{application.job.role}</td>
                    <td>{application.job.company}</td>
                    <td>
                      {application.job.match_score === null
                        ? "—"
                        : `${Math.round(application.job.match_score * 100)}%`}
                    </td>
                    <td>
                      <span className={`badge ${STATE_TONE[application.state] ?? ""}`}>
                        {application.state}
                      </span>
                      {application.dry_run && (
                        <span className="badge accent" style={{ marginLeft: 6 }}>
                          dry run
                        </span>
                      )}
                    </td>
                    <td className="muted">
                      {new Date(application.started_at).toLocaleString()}
                    </td>
                    <td>
                      <button
                        className="secondary small"
                        onClick={() =>
                          setExpanded(expanded === application.id ? null : application.id)
                        }
                      >
                        {expanded === application.id ? "Hide" : "Details"}
                      </button>
                    </td>
                  </tr>
                  {expanded === application.id && (
                    <tr>
                      <td colSpan={6} style={{ background: "var(--bg)" }}>
                        <Detail application={application} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function Detail({ application }: { application: ApplicationSummary }) {
  const report = application.job.match_report as
    | {
        missing_required?: string[];
        required_satisfied?: string[];
        risks?: string[];
        advantages?: string[];
        summary?: string;
        rejected_reason?: string;
      }
    | null;
  const evidence = application.submission_evidence as
    | { confirmed?: boolean; signals?: string[]; reference?: string }
    | null;

  return (
    <div style={{ padding: "6px 0" }}>
      {report?.summary && <p style={{ marginTop: 0 }}>{report.summary}</p>}
      {report?.rejected_reason && (
        <p className="muted">Rejected: {report.rejected_reason}</p>
      )}

      <div className="grid cols-2">
        {report?.required_satisfied?.length ? (
          <div>
            <strong>Requirements met</strong>
            <ul>
              {report.required_satisfied.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {report?.missing_required?.length ? (
          <div>
            <strong>Missing requirements</strong>
            <ul>
              {report.missing_required.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

      {application.checkpoints.length > 0 && (
        <p>
          <strong>Checkpoints:</strong>{" "}
          <span className="pill-list" style={{ display: "inline-flex" }}>
            {application.checkpoints.map((c) => (
              <span className="badge ok" key={c}>
                {c}
              </span>
            ))}
          </span>
        </p>
      )}

      {evidence && (
        <p>
          <strong>Submission evidence:</strong>{" "}
          {evidence.confirmed ? (
            <span className="badge ok">confirmed</span>
          ) : (
            <span className="badge danger">not confirmed</span>
          )}
          {evidence.reference && <span className="mono"> ref {evidence.reference}</span>}
          {evidence.signals?.length ? (
            <span className="muted"> — {evidence.signals.join("; ")}</span>
          ) : null}
        </p>
      )}

      {application.errors.length > 0 && (
        <div className="banner danger" style={{ marginTop: 10 }}>
          {application.errors.join(" · ")}
        </div>
      )}

      {application.resume_version_id && (
        <p className="muted mono">
          Resume version {application.resume_version_id} — the file may have been deleted,
          but the recipe is retained and can rebuild it.
        </p>
      )}
    </div>
  );
}
