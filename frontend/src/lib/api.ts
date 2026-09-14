/**
 * Typed client for the CVnator API.
 *
 * Every call goes through `request`, so a failure surfaces the server's own
 * explanation rather than a generic "fetch failed" — the API returns real reasons
 * (Ollama not running, no CV uploaded, a session already active) and the UI is far
 * more useful when it shows them verbatim.
 */

export interface SessionMetrics {
  jobs_discovered: number;
  jobs_analyzed: number;
  jobs_rejected: number;
  jobs_matched: number;
  applications_started: number;
  applications_completed: number;
  applications_failed: number;
  applications_waiting_user: number;
  resumes_generated: number;
  agent_steps_total: number;
  tool_retries_total: number;
  vision_fallbacks: number;
  human_interventions: number;
  invalid_answers_blocked: number;
  execution_seconds: number;
}

export interface CurrentActivity {
  stage: string;
  job_title: string;
  company: string;
  platform: string;
  match_percent: number | null;
  ats_percent: number | null;
  step: string;
}

export type SessionStatus =
  | "stopped"
  | "running"
  | "paused"
  | "user_control"
  | "stopping";

export interface SessionSnapshot {
  status: SessionStatus;
  metrics: Partial<SessionMetrics>;
  activity: Partial<CurrentActivity>;
  queues: Record<string, number>;
  autonomy: AutonomySettings;
  browser_user_control?: boolean;
  applications_today?: number;
}

export interface AutonomySettings {
  auto_search: boolean;
  auto_match: boolean;
  auto_optimize_resume: boolean;
  auto_apply: boolean;
  auto_submit: boolean;
  auto_create_external_account: boolean;
  account_policy: "never" | "ask" | "auto";
  delete_temp_resumes: boolean;
  dry_run: boolean;
  review_before_submit: boolean;
  minimum_match_score: number;
  daily_application_limit: number;
  session_application_limit: number;
}

export interface SearchProfile {
  id?: number | null;
  name: string;
  enabled: boolean;
  role_title: string;
  alternate_titles: string[];
  role_description: string;
  keywords: string[];
  required_keywords: string[];
  excluded_keywords: string[];
  technologies: string[];
  location: string;
  remote_policies: string[];
  seniorities: string[];
  salary_currency: string;
  salary_minimum: number | null;
  languages: string[];
  companies_whitelist: string[];
  companies_blacklist: string[];
  platforms: string[];
  minimum_match_score: number;
  daily_application_limit: number;
  priority: number;
}

export interface ApplicationSummary {
  id: number;
  state: string;
  platform: string;
  external_platform: string;
  dry_run: boolean;
  started_at: string;
  completed_at: string | null;
  checkpoints: string[];
  errors: string[];
  resume_version_id: string;
  submission_evidence: Record<string, unknown> | null;
  job: {
    id: number;
    role: string;
    company: string;
    location: string;
    url: string;
    match_score: number | null;
    match_report: Record<string, unknown> | null;
  };
}

export interface UserAction {
  id: number;
  kind: string;
  platform: string;
  company: string;
  job_title: string;
  problem: string;
  action_needed: string;
  question: string;
  resolved: boolean;
  created_at: string;
}

export interface LogEntry {
  ts: number;
  level: string;
  logger: string;
  message: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: init?.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* the body was not JSON; the status text is the best we have */
    }
    throw new ApiError(detail, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Record<string, unknown>>("/health"),

  getProfile: () =>
    request<{ exists: boolean; profile: Record<string, unknown> | null }>("/profile"),
  uploadCv: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ profile: Record<string, unknown>; master_cv_path: string }>(
      "/profile/upload",
      { method: "POST", body: form },
    );
  },
  saveProfile: (profile: unknown) =>
    request<{ ok: boolean; version: number }>("/profile", {
      method: "PUT",
      body: JSON.stringify(profile),
    }),

  listSearches: () => request<SearchProfile[]>("/searches"),
  createSearch: (search: SearchProfile) =>
    request<SearchProfile>("/searches", { method: "POST", body: JSON.stringify(search) }),
  updateSearch: (id: number, search: SearchProfile) =>
    request<SearchProfile>(`/searches/${id}`, { method: "PUT", body: JSON.stringify(search) }),
  deleteSearch: (id: number) =>
    request<{ deleted: boolean }>(`/searches/${id}`, { method: "DELETE" }),

  getSession: () => request<SessionSnapshot>("/session"),
  startSession: () => request<SessionSnapshot>("/session/start", { method: "POST" }),
  pauseSession: () => request<SessionSnapshot>("/session/pause", { method: "POST" }),
  resumeSession: () => request<SessionSnapshot>("/session/resume", { method: "POST" }),
  stopSession: () => request<SessionSnapshot>("/session/stop", { method: "POST" }),
  skipJob: () => request<SessionSnapshot>("/session/skip", { method: "POST" }),
  takeControl: () => request<SessionSnapshot>("/session/take-control", { method: "POST" }),
  returnControl: () => request<SessionSnapshot>("/session/return-control", { method: "POST" }),

  getAutonomy: () => request<AutonomySettings>("/settings/autonomy"),
  setAutonomy: (settings: AutonomySettings) =>
    request<AutonomySettings>("/settings/autonomy", {
      method: "PUT",
      body: JSON.stringify(settings),
    }),

  listApplications: (limit = 100) =>
    request<ApplicationSummary[]>(`/applications?limit=${limit}`),
  listJobs: (limit = 200) => request<unknown[]>(`/jobs?limit=${limit}`),

  listUserActions: () => request<UserAction[]>("/user-actions"),
  answerUserAction: (id: number, answer: string | null) =>
    request<{ delivered: boolean }>(`/user-actions/${id}/answer`, {
      method: "POST",
      body: JSON.stringify({ answer }),
    }),

  listMemory: () => request<Record<string, unknown>[]>("/memory"),
  deleteMemory: (id: number) =>
    request<{ deleted: boolean }>(`/memory/${id}`, { method: "DELETE" }),

  storage: () => request<{ usage: Record<string, { count: number; bytes: number }> }>("/storage"),
  collectStorage: (policy: string) =>
    request<{ deleted: number; freed_bytes: number }>(`/storage/collect?policy=${policy}`, {
      method: "POST",
    }),

  logs: (limit = 200, levels?: string[]) =>
    request<LogEntry[]>(
      `/logs?limit=${limit}${levels?.length ? `&levels=${levels.join(",")}` : ""}`,
    ),
};

export function emptySearch(): SearchProfile {
  return {
    name: "",
    enabled: true,
    role_title: "",
    alternate_titles: [],
    role_description: "",
    keywords: [],
    required_keywords: [],
    excluded_keywords: [],
    technologies: [],
    location: "",
    remote_policies: ["any"],
    seniorities: [],
    salary_currency: "BRL",
    salary_minimum: null,
    languages: [],
    companies_whitelist: [],
    companies_blacklist: [],
    platforms: ["mock"],
    minimum_match_score: 0.7,
    daily_application_limit: 10,
    priority: 0,
  };
}
