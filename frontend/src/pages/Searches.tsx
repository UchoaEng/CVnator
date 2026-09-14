import { useEffect, useState } from "react";
import { api, emptySearch, type SearchProfile } from "../lib/api";

const PLATFORMS = ["mock", "generic"];
const REMOTE = ["any", "remote", "hybrid", "onsite"];

function ListField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: string[];
  onChange: (next: string[]) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="text"
        value={value.join(", ")}
        onChange={(e) =>
          onChange(
            e.target.value
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean),
          )
        }
      />
      {hint && <small>{hint}</small>}
    </label>
  );
}

export function SearchesPage() {
  const [searches, setSearches] = useState<SearchProfile[]>([]);
  const [draft, setDraft] = useState<SearchProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = () =>
    api
      .listSearches()
      .then(setSearches)
      .catch((e) => setError(String(e.message ?? e)));

  useEffect(() => {
    void reload();
  }, []);

  async function save() {
    if (!draft) return;
    try {
      if (draft.id) await api.updateSearch(draft.id, draft);
      else await api.createSearch(draft);
      setDraft(null);
      setError(null);
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function toggle(search: SearchProfile) {
    if (!search.id) return;
    await api.updateSearch(search.id, { ...search, enabled: !search.enabled });
    await reload();
  }

  async function remove(id: number) {
    await api.deleteSearch(id);
    await reload();
  }

  return (
    <>
      <h1>Job Searches</h1>
      <p className="page-sub">
        Each search runs independently. Keyword and company rules here are enforced
        literally — the model never gets to make an exception to them.
      </p>

      {error && <div className="banner danger">{error}</div>}

      {!draft && (
        <div className="row" style={{ marginBottom: 16 }}>
          <button onClick={() => setDraft(emptySearch())}>New search</button>
        </div>
      )}

      {draft && (
        <div className="card">
          <h2>{draft.id ? "Edit search" : "New search"}</h2>
          <div className="grid cols-2">
            <label className="field">
              <span>Name</span>
              <input
                type="text"
                value={draft.name}
                onChange={(e) => setDraft({ ...draft, name: e.target.value })}
              />
            </label>
            <label className="field">
              <span>Role title</span>
              <input
                type="text"
                value={draft.role_title}
                onChange={(e) => setDraft({ ...draft, role_title: e.target.value })}
              />
            </label>
          </div>

          <ListField
            label="Alternate titles"
            hint="Comma separated. Helps match postings that name the role differently."
            value={draft.alternate_titles}
            onChange={(v) => setDraft({ ...draft, alternate_titles: v })}
          />
          <ListField
            label="Keywords"
            value={draft.keywords}
            onChange={(v) => setDraft({ ...draft, keywords: v })}
          />
          <ListField
            label="Required keywords"
            hint="A posting missing any of these is rejected outright."
            value={draft.required_keywords}
            onChange={(v) => setDraft({ ...draft, required_keywords: v })}
          />
          <ListField
            label="Excluded keywords"
            hint="A posting containing any of these is rejected outright."
            value={draft.excluded_keywords}
            onChange={(v) => setDraft({ ...draft, excluded_keywords: v })}
          />
          <ListField
            label="Technologies"
            value={draft.technologies}
            onChange={(v) => setDraft({ ...draft, technologies: v })}
          />

          <div className="grid cols-2">
            <label className="field">
              <span>Location</span>
              <input
                type="text"
                value={draft.location}
                onChange={(e) => setDraft({ ...draft, location: e.target.value })}
              />
            </label>
            <label className="field">
              <span>Minimum salary</span>
              <input
                type="number"
                value={draft.salary_minimum ?? ""}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    salary_minimum: e.target.value ? Number(e.target.value) : null,
                  })
                }
              />
            </label>
          </div>

          <label className="field">
            <span>Work arrangement</span>
            <div className="pill-list">
              {REMOTE.map((policy) => (
                <label className="check" key={policy}>
                  <input
                    type="checkbox"
                    checked={draft.remote_policies.includes(policy)}
                    onChange={(e) =>
                      setDraft({
                        ...draft,
                        remote_policies: e.target.checked
                          ? [...draft.remote_policies, policy]
                          : draft.remote_policies.filter((p) => p !== policy),
                      })
                    }
                  />
                  {policy}
                </label>
              ))}
            </div>
          </label>

          <label className="field">
            <span>Platforms</span>
            <div className="pill-list">
              {PLATFORMS.map((platform) => (
                <label className="check" key={platform}>
                  <input
                    type="checkbox"
                    checked={draft.platforms.includes(platform)}
                    onChange={(e) =>
                      setDraft({
                        ...draft,
                        platforms: e.target.checked
                          ? [...draft.platforms, platform]
                          : draft.platforms.filter((p) => p !== platform),
                      })
                    }
                  />
                  {platform}
                </label>
              ))}
            </div>
          </label>

          <ListField
            label="Company blacklist"
            value={draft.companies_blacklist}
            onChange={(v) => setDraft({ ...draft, companies_blacklist: v })}
          />
          <ListField
            label="Company whitelist"
            hint="If set, only these companies are considered."
            value={draft.companies_whitelist}
            onChange={(v) => setDraft({ ...draft, companies_whitelist: v })}
          />

          <div className="grid cols-3">
            <label className="field">
              <span>Minimum match ({Math.round(draft.minimum_match_score * 100)}%)</span>
              <input
                type="number"
                min={0}
                max={100}
                value={Math.round(draft.minimum_match_score * 100)}
                onChange={(e) =>
                  setDraft({ ...draft, minimum_match_score: Number(e.target.value) / 100 })
                }
              />
            </label>
            <label className="field">
              <span>Daily application limit</span>
              <input
                type="number"
                min={0}
                value={draft.daily_application_limit}
                onChange={(e) =>
                  setDraft({ ...draft, daily_application_limit: Number(e.target.value) })
                }
              />
            </label>
            <label className="field">
              <span>Priority (0–10)</span>
              <input
                type="number"
                min={0}
                max={10}
                value={draft.priority}
                onChange={(e) => setDraft({ ...draft, priority: Number(e.target.value) })}
              />
            </label>
          </div>

          <div className="row">
            <button onClick={() => void save()}>Save</button>
            <button className="secondary" onClick={() => setDraft(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {searches.length === 0 && !draft && (
        <div className="card empty">No searches yet. Create one to tell the agent what to look for.</div>
      )}

      {searches.map((search) => (
        <div className="card" key={search.id}>
          <div className="spread">
            <div>
              <strong>{search.name || search.role_title}</strong>
              <div className="muted">
                {search.role_title} · {search.location || "any location"} ·{" "}
                {search.platforms.join(", ")} · min {Math.round(search.minimum_match_score * 100)}%
              </div>
            </div>
            <div className="row">
              <span className={`badge ${search.enabled ? "ok" : ""}`}>
                {search.enabled ? "enabled" : "disabled"}
              </span>
              <button className="secondary small" onClick={() => void toggle(search)}>
                {search.enabled ? "Disable" : "Enable"}
              </button>
              <button className="secondary small" onClick={() => setDraft(search)}>
                Edit
              </button>
              <button
                className="danger small"
                onClick={() => search.id && void remove(search.id)}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      ))}
    </>
  );
}
