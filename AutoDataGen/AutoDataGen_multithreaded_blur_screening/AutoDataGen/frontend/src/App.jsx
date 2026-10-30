import { useEffect, useRef, useState } from "react";
import { Database } from "lucide-react";
import DatasetSelectionPage from "./pages/DatasetSelectionPage.jsx";
import TextAugmentationPage from "./pages/TextAugmentationPage.jsx";
import TableAugmentationPage from "./pages/TableAugmentationPage.jsx";
import ImageAugmentationPage from "./pages/ImageAugmentationPage.jsx";
import { DATASET_TYPES } from "./lib/datasetTypes.js";
import { api } from "./lib/api.js";

const pages = {
  "/": DatasetSelectionPage,
  "/augment/text": TextAugmentationPage,
  "/augment/table": TableAugmentationPage,
  "/augment/image": ImageAugmentationPage,
};

function currentPath() {
  const path = window.location.hash.slice(1).split("?")[0].replace(/\/+$/, "");
  return path || "/";
}

export default function App() {
  const [path, setPath] = useState(currentPath);
  const [status, setStatus] = useState(null);
  const [checking, setChecking] = useState(false);
  const statusController = useRef(null);
  const Page = pages[path];
  const selected = DATASET_TYPES.find((type) => type.href === `#${path}`);

  useEffect(() => {
    const onHashChange = () => setPath(currentPath());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    document.title = selected
      ? `${selected.title} | AutoDataGen`
      : "Choose dataset type | AutoDataGen";
    window.scrollTo({ top: 0, behavior: "instant" });
    document.getElementById("main-content")?.focus({ preventScroll: true });
  }, [path, selected]);

  async function checkStatus() {
    statusController.current?.abort();
    const controller = new AbortController();
    statusController.current = controller;
    setChecking(true);
    try {
      const { data } = await api.get("/api/ollama/status", {
        timeout: 5000,
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setStatus(data);
    } catch {
      if (!controller.signal.aborted)
        setStatus({
          available: false,
          message:
            "Could not reach the backend. Start the FastAPI server and check the API connection.",
        });
    } finally {
      if (!controller.signal.aborted) setChecking(false);
    }
  }

  useEffect(() => {
    checkStatus();
    return () => statusController.current?.abort();
  }, []);

  const modelReady = status?.available && status?.model_available;

  return (
    <div className="shell">
      <a
        className="skip-link"
        href="#main-content"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main-content")?.focus();
        }}
      >
        Skip to content
      </a>
      <header>
        <a className="brand" href="#/" aria-label="AutoDataGen dataset chooser">
          <Database aria-hidden="true" />
          <b>AutoDataGen</b>
        </a>
        <div className="header-status">
          <span
            className={
              modelReady ? "service-status connected" : "service-status"
            }
            title={status?.message || "Checking local model"}
            role="status"
          >
            Local model:{" "}
            {checking
              ? "checking"
              : modelReady
                ? "connected"
                : status?.enabled === false
                  ? "disabled"
                  : "not ready"}
          </span>
          <button
            className="secondary small-button"
            onClick={checkStatus}
            disabled={checking}
          >
            Check connection
          </button>
        </div>
      </header>
      <main id="main-content" tabIndex={-1}>
        {selected && (
          <nav className="workspace-nav" aria-label="Dataset workspaces">
            <a href="#/">Dataset types</a>
            <span className="nav-divider" aria-hidden="true">
              /
            </span>
            {DATASET_TYPES.map((type) => (
              <a
                href={type.href}
                key={type.id}
                aria-current={type.id === selected.id ? "page" : undefined}
              >
                {type.label}
              </a>
            ))}
          </nav>
        )}
        {Page ? (
          <Page key={path} />
        ) : (
          <section className="hero">
            <small>PAGE NOT FOUND</small>
            <h1>Choose a dataset type</h1>
            <p>This workspace link is not available.</p>
            <a className="button" href="#/">
              Open dataset chooser
            </a>
          </section>
        )}
        <footer>
          <span>AutoDataGen · Dataset augmentation workspace</span>
          {selected && <a href="#/">Change dataset type</a>}
        </footer>
      </main>
    </div>
  );
}
