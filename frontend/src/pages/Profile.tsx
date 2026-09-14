import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";

interface Experience {
  company: string;
  job_title: string;
  start_date: string;
  end_date: string;
  is_current: boolean;
  location: string;
  responsibilities: string[];
  achievements: string[];
  technologies: string[];
}

interface Profile {
  personal_information: Record<string, string>;
  professional_summary: string;
  headline: string;
  experiences: Experience[];
  skills: { name: string; category: string; years_of_experience: number | null }[];
  languages: { language: string; proficiency: string }[];
  education: Record<string, string>[];
  certifications: Record<string, string>[];
  work_authorization: { countries_authorized: string[]; visa_status: string };
  salary_preferences: { currency: string; minimum: number | null; target: number | null };
  [key: string]: unknown;
}

export function ProfilePage() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .getProfile()
      .then((r) => setProfile(r.profile as Profile | null))
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  async function upload(file: File) {
    setBusy(true);
    setError(null);
    setStatus("Extracting text and parsing the CV with the local model — this can take a minute.");
    try {
      const result = await api.uploadCv(file);
      setProfile(result.profile as Profile);
      setStatus("CV parsed. Review the extracted facts below — everything the agent says comes from here.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    if (!profile) return;
    setBusy(true);
    try {
      const result = await api.saveProfile(profile);
      setStatus(`Saved as profile version ${result.version}.`);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function patchPersonal(key: string, value: string) {
    if (!profile) return;
    setProfile({
      ...profile,
      personal_information: { ...profile.personal_information, [key]: value },
    });
  }

  return (
    <>
      <h1>Profile</h1>
      <p className="page-sub">
        Your master CV becomes a structured profile. It is the only source of facts the
        agent may state — anything not here, it will ask you about rather than guess.
      </p>

      {error && <div className="banner danger">{error}</div>}
      {status && <div className="banner info">{status}</div>}

      <div className="card">
        <h2>Master CV</h2>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.docx,.txt,.md"
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void upload(file);
          }}
        />
        <div className="row">
          <button disabled={busy} onClick={() => fileRef.current?.click()}>
            {profile ? "Replace CV" : "Upload CV"}
          </button>
          <span className="muted">
            {profile
              ? "Your original file is kept permanently and is never deleted by cleanup."
              : "PDF or DOCX. Scanned images are not supported — the text must be selectable."}
          </span>
        </div>
      </div>

      {!profile && !busy && (
        <div className="card empty">Upload a CV to get started.</div>
      )}

      {profile && (
        <>
          <div className="card">
            <h2>Personal information</h2>
            <div className="grid cols-2">
              {["full_name", "email", "phone", "city", "state", "country", "linkedin", "github"].map(
                (key) => (
                  <label className="field" key={key}>
                    <span>{key.replace(/_/g, " ")}</span>
                    <input
                      type="text"
                      value={profile.personal_information[key] ?? ""}
                      onChange={(e) => patchPersonal(key, e.target.value)}
                    />
                  </label>
                ),
              )}
            </div>
          </div>

          <div className="card">
            <h2>Summary</h2>
            <label className="field">
              <span>Headline</span>
              <input
                type="text"
                value={profile.headline}
                onChange={(e) => setProfile({ ...profile, headline: e.target.value })}
              />
            </label>
            <label className="field">
              <span>Professional summary</span>
              <textarea
                value={profile.professional_summary}
                onChange={(e) =>
                  setProfile({ ...profile, professional_summary: e.target.value })
                }
              />
            </label>
          </div>

          <div className="card">
            <h2>Experience ({profile.experiences.length})</h2>
            {profile.experiences.map((exp, index) => (
              <div
                key={index}
                style={{
                  borderTop: index ? "1px solid var(--border)" : undefined,
                  paddingTop: index ? 14 : 0,
                  marginTop: index ? 14 : 0,
                }}
              >
                <div className="spread">
                  <strong>
                    {exp.job_title} — {exp.company}
                  </strong>
                  <span className="muted">
                    {exp.start_date} → {exp.is_current ? "present" : exp.end_date || "?"}
                  </span>
                </div>
                <ul style={{ margin: "8px 0", paddingLeft: 18 }}>
                  {[...exp.responsibilities, ...exp.achievements].map((bullet, i) => (
                    <li key={i}>{bullet}</li>
                  ))}
                </ul>
                <div className="pill-list">
                  {exp.technologies.map((tech) => (
                    <span className="badge" key={tech}>
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            {profile.experiences.length === 0 && (
              <div className="muted">No experience was extracted. Check the CV text.</div>
            )}
          </div>

          <div className="card">
            <h2>Skills ({profile.skills.length})</h2>
            <div className="pill-list">
              {profile.skills.map((skill) => (
                <span className="badge accent" key={skill.name}>
                  {skill.name}
                  {skill.years_of_experience ? ` · ${skill.years_of_experience}y` : ""}
                </span>
              ))}
            </div>
          </div>

          <div className="card">
            <h2>Languages & authorization</h2>
            <div className="pill-list" style={{ marginBottom: 12 }}>
              {profile.languages.map((lang) => (
                <span className="badge" key={lang.language}>
                  {lang.language} · {lang.proficiency}
                </span>
              ))}
            </div>
            <p className="muted" style={{ margin: 0 }}>
              Work authorization:{" "}
              {profile.work_authorization.countries_authorized.length
                ? profile.work_authorization.countries_authorized.join(", ")
                : "not recorded — the agent will ask you rather than assume"}
            </p>
          </div>

          <div className="row">
            <button disabled={busy} onClick={() => void save()}>
              Save profile
            </button>
            <span className="muted">
              Saving creates a new version; generated CVs record which version they came from.
            </span>
          </div>
        </>
      )}
    </>
  );
}
