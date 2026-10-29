import axios from "axios";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export const api = axios.create({ baseURL: API_BASE, timeout: 30000 });
export const apiUrl = (path) => `${API_BASE}${path}`;
export const isActiveRun = (run) =>
  ["queued", "processing", "finalizing"].includes(run?.status);

export function errorMessage(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join("; ");
  if (error.code === "ECONNABORTED") {
    return "The backend took too long to respond. Check that it is running and try again.";
  }
  return error.message || "The request failed. Please try again.";
}

export function validateFile(file, extensions) {
  if (!file) return "";
  if (
    !extensions.some((extension) => file.name.toLowerCase().endsWith(extension))
  ) {
    return `Choose a supported file: ${extensions.join(", ")}.`;
  }
  if (file.size === 0)
    return "The selected file is empty. Choose a dataset with content.";
  return "";
}
