import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Shell } from "./components/Shell";
import { ActionsPage } from "./pages/Actions";
import { ApplicationsPage } from "./pages/Applications";
import { Dashboard } from "./pages/Dashboard";
import { ProfilePage } from "./pages/Profile";
import { SearchesPage } from "./pages/Searches";
import { SettingsPage } from "./pages/Settings";
import { useSession } from "./lib/useSession";

export function App() {
  const { snapshot, actions, logs, connected, error, refresh, control } = useSession();

  return (
    <BrowserRouter>
      <Routes>
        <Route
          element={<Shell snapshot={snapshot} actions={actions} connected={connected} />}
        >
          <Route
            index
            element={
              <Dashboard
                snapshot={snapshot}
                actions={actions}
                logs={logs}
                error={error}
                control={control}
              />
            }
          />
          <Route path="profile" element={<ProfilePage />} />
          <Route path="searches" element={<SearchesPage />} />
          <Route path="applications" element={<ApplicationsPage />} />
          <Route
            path="actions"
            element={<ActionsPage actions={actions} onResolved={() => void refresh()} />}
          />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
