import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type LogEntry,
  type SessionSnapshot,
  type UserAction,
} from "./api";

/**
 * Live session state.
 *
 * The server pushes over SSE; polling exists only as the fallback path for when
 * the stream is not connected, so a dropped EventSource degrades to a slower UI
 * rather than a frozen one.
 */
export function useSession() {
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [actions, setActions] = useState<UserAction[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [session, pending] = await Promise.all([
        api.getSession(),
        api.listUserActions(),
      ]);
      setSnapshot(session);
      setActions(pending);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const refreshLogs = useCallback(async () => {
    try {
      setLogs(await api.logs(200));
    } catch {
      /* logs are informational; a failure here must not break the page */
    }
  }, []);

  useEffect(() => {
    void refresh();
    void refreshLogs();

    const source = new EventSource("/api/events");
    sourceRef.current = source;
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);

    // Any event that changes what the dashboard shows triggers one refresh.
    const wake = () => {
      void refresh();
    };
    for (const kind of [
      "session",
      "metrics",
      "activity",
      "application_state",
      "application_submitted",
      "application_waiting",
      "user_action_required",
      "resume_generated",
      "artifacts_cleaned",
      "autonomy_updated",
    ]) {
      source.addEventListener(kind, wake);
    }

    // Slow poll: covers a dropped stream and keeps the log tail moving.
    const timer = window.setInterval(() => {
      void refreshLogs();
      if (!sourceRef.current || sourceRef.current.readyState !== EventSource.OPEN) {
        void refresh();
      }
    }, 4000);

    return () => {
      source.close();
      sourceRef.current = null;
      window.clearInterval(timer);
    };
  }, [refresh, refreshLogs]);

  const control = useCallback(
    async (action: () => Promise<SessionSnapshot>) => {
      try {
        setSnapshot(await action());
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
      void refresh();
    },
    [refresh],
  );

  return { snapshot, actions, logs, connected, error, refresh, control, setError };
}
