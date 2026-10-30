import { useEffect, useRef, useState } from "react";
import { api, errorMessage, isActiveRun } from "../lib/api.js";

function storedRun(key) {
  try {
    return sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

function rememberRun(key, id) {
  try {
    if (id) sessionStorage.setItem(key, id);
    else sessionStorage.removeItem(key);
  } catch {
    /* The workspace also works when browser storage is disabled. */
  }
}

export default function useAugmentationRun(type) {
  const storageKey = `autodatagen:${type}:last-run`;
  const [runId, setRunId] = useState(() => storedRun(storageKey));
  const [run, setRun] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [restoring, setRestoring] = useState(() =>
    Boolean(storedRun(storageKey)),
  );
  const [error, setError] = useState("");
  const [pollError, setPollError] = useState("");
  const [retryKey, setRetryKey] = useState(0);
  const [cancelRequested, setCancelRequested] = useState(false);
  const requestController = useRef(null);

  useEffect(() => () => requestController.current?.abort(), []);

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    let timer;
    let failures = 0;

    async function poll() {
      try {
        const response = await api.get(`/api/runs/${runId}`, {
          signal: controller.signal,
          timeout: 10000,
        });
        if (controller.signal.aborted) return;
        failures = 0;
        setRun(response.data);
        setRestoring(false);
        setPollError("");
        if (isActiveRun(response.data)) timer = setTimeout(poll, 2000);
        else setCancelRequested(false);
      } catch (failure) {
        if (controller.signal.aborted) return;
        if (failure.response?.status === 404) {
          rememberRun(storageKey, null);
          setRunId(null);
          setRun(null);
          setRestoring(false);
          setError(
            "The previous run is no longer available. Select a dataset to start again.",
          );
        } else if (++failures < 3) {
          timer = setTimeout(poll, 1500);
        } else {
          setPollError(
            "Cannot read the run status. Check the backend connection, then resume status updates. The backend job may still be running.",
          );
        }
      }
    }

    poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [runId, retryKey, storageKey]);

  async function startRun(form) {
    const controller = new AbortController();
    requestController.current = controller;
    setSubmitting(true);
    setError("");
    setPollError("");
    setCancelRequested(false);
    try {
      const response = await api.post("/api/runs", form, {
        timeout: 30000,
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      setRun(response.data);
      rememberRun(storageKey, response.data.run_id);
      setRunId(response.data.run_id);
    } catch (failure) {
      if (!controller.signal.aborted) setError(errorMessage(failure));
    } finally {
      if (!controller.signal.aborted) setSubmitting(false);
    }
  }

  async function cancelRun() {
    if (!runId || cancelRequested) return;
    const controller = new AbortController();
    requestController.current = controller;
    setCancelRequested(true);
    try {
      await api.post(`/api/runs/${runId}/cancel`, null, {
        signal: controller.signal,
      });
    } catch (failure) {
      if (!controller.signal.aborted) {
        setCancelRequested(false);
        setError(errorMessage(failure));
      }
    }
  }

  function resetRun() {
    rememberRun(storageKey, null);
    setRunId(null);
    setRun(null);
    setRestoring(false);
    setError("");
    setPollError("");
    setCancelRequested(false);
  }

  return {
    run,
    startRun,
    resetRun,
    cancelRun,
    cancelRequested,
    error,
    pollError,
    restoring,
    busy: submitting || restoring || isActiveRun(run),
    resumePolling: () => {
      setPollError("");
      setRetryKey((key) => key + 1);
    },
  };
}
