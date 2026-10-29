# Dataset chooser and separate workspaces

Only the AutoDataGen project was changed. The separately uploaded frontend ZIP is not needed.

## Start the application

Start the backend using the existing commands in README.md. In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/`. Choose **Text**, **Table**, or **Image**. Each workspace offers **Change dataset type**, navigation to the other workspaces, analysis, its own settings, results, and downloads.

## Routes

| Page | URL | Component |
| --- | --- | --- |
| Dataset chooser | `http://127.0.0.1:5173/#/` | `src/pages/DatasetSelectionPage.jsx` |
| Text augmentation | `http://127.0.0.1:5173/#/augment/text` | `src/pages/TextAugmentationPage.jsx` |
| Table augmentation | `http://127.0.0.1:5173/#/augment/table` | `src/pages/TableAugmentationPage.jsx` |
| Image augmentation | `http://127.0.0.1:5173/#/augment/image` | `src/pages/ImageAugmentationPage.jsx` |

These are separate React pages using hash routes. Direct links, refresh, and browser Back/Forward do not need server-side route rewrites. An invalid route shows a recovery link to the chooser.

## Existing login or external links

This project ZIP has no authentication endpoints or login screen. Its main URL opens the chooser directly. If you already have a separate login page, set its successful-login redirect and its Start/Generate button link to the AutoDataGen origin, for example:

```js
window.location.assign("http://127.0.0.1:5173/#/");
```

Use your actual AutoDataGen URL when hosted. A redirect opens the chooser; it does not add authentication or protect the backend API.

## Workspace behavior

- **Text:** TXT with one non-empty example per line, or CSV/XLSX/XLSM with text. Select the label column for supervised text datasets; select the text column for an unlabeled corpus. Single-column CSV/Excel text corpora also use asynchronous text-only augmentation, even for short or repeated values. Unlabeled text reports generated counts and examples without classification accuracy. Text generation requires the existing local model integration.
- **Table:** CSV/XLSX/XLSM with feature columns and a target column. Supports the existing classification/regression search, iteration limits up to 60, progress polling, cancellation, metrics, and dataset/report exports. Plain TXT and image ZIPs are not accepted on this page.
- **Image:** ZIP with class folders, YOLO labels, or paired segmentation masks. Analyze first, choose the detected task or correct it, set execution mode and variants, then run augmentation. The page retains generated previews, annotation checks, model evaluation scope, and the downloadable ZIP.

Text and table runs are isolated by workspace. The last run ID for each is remembered in the current browser tab; returning to the page or refreshing restores progress/results from the backend. Changing pages stops frontend polling, while the backend job continues. File selections are not stored by the browser. Image requests retain their existing synchronous backend behavior; keep that page open until the export is ready.

## API configuration

`src/lib/api.js` centralizes API requests and download URLs. By default, development and preview proxy `/api` to `http://127.0.0.1:8000`. Optional settings in `frontend/.env.example`:

```dotenv
VITE_API_BASE_URL=
AUTODATAGEN_API_PROXY_TARGET=http://127.0.0.1:8000
```

For a hosted frontend, serve `/api` through a reverse proxy to FastAPI, or set `VITE_API_BASE_URL` to a reachable backend origin and allow that frontend origin in backend CORS. Rebuild after changing `VITE_API_BASE_URL`.

## Shared implementation

- `src/components/DataAugmentationWorkspace.jsx`: shared text/table workflow UI with type-specific uploads and analysis.
- `src/hooks/useAugmentationRun.js`: asynchronous submission, progress, cancellation, reconnect, and run restoration.
- `src/components/DatasetProfile.jsx`: separate handling for text corpus and structured-data analysis.
- `src/components/RunReport.jsx`: existing text, classification, regression, and iteration reports.
- `src/components/ImageReport.jsx`: existing image report, previews, and evaluation limitations.

The existing backend algorithms, validation-based model selection, 60-iteration cap, and image evaluation boundaries remain in place. The small backend compatibility change allows single-column CSV/Excel text corpora to use the same asynchronous workflow as TXT.
