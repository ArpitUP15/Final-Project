import Metric from "./Metric.jsx";

export default function DatasetProfile({ analysis, columns, preview }) {
  const pureText = analysis.task_type === "text_corpus";
  const lengths = analysis.document_length_characters;
  return (
    <>
      <div className="metrics">
        <Metric label="Rows" value={analysis.shape?.rows} />
        <Metric label="Columns" value={analysis.shape?.columns} />
        {pureText ? (
          <>
            <Metric
              label="Non-empty documents"
              value={analysis.non_empty_documents}
            />
            <Metric label="Empty documents" value={analysis.empty_documents} />
            <Metric
              label="Unique documents"
              value={
                analysis.unique_document_ratio == null
                  ? "N/A"
                  : `${(analysis.unique_document_ratio * 100).toFixed(1)}%`
              }
            />
          </>
        ) : (
          <>
            <Metric
              label="Missing cells"
              value={analysis.missing_values?.total_missing_cells}
            />
            <Metric label="Duplicate rows" value={analysis.duplicate_rows} />
            <Metric
              label="Quality heuristic"
              value={
                analysis.quality_score == null
                  ? "N/A"
                  : `${analysis.quality_score}/100`
              }
            />
          </>
        )}
      </div>
      {analysis.summary && <p className="notice">{analysis.summary}</p>}
      <div className="twocol">
        <div className="card">
          <h3>{pureText ? "Text profile" : "Feature profile"}</h3>
          {pureText ? (
            <>
              <p>Text column: {analysis.text_column}</p>
              <p>Average characters: {lengths?.mean ?? "N/A"}</p>
              <p>
                Shortest / longest document: {lengths?.min ?? "N/A"} /{" "}
                {lengths?.max ?? "N/A"}
              </p>
              <small>
                Text without labels is augmented without calculating
                classification accuracy.
              </small>
            </>
          ) : (
            <>
              <p>
                Numeric columns: {analysis.column_types?.numeric?.length ?? 0}
              </p>
              <p>
                Categorical columns:{" "}
                {analysis.column_types?.categorical?.length ?? 0}
              </p>
              {analysis.text_column_names?.length > 0 && (
                <p>Text columns: {analysis.text_column_names.join(", ")}</p>
              )}
              <h3>Missing values</h3>
              {Object.entries(analysis.missing_values?.by_column || {})
                .filter(([, value]) => value > 0)
                .map(([key, value]) => (
                  <p key={key}>
                    {key}: {value}
                  </p>
                ))}
              {analysis.missing_values?.total_missing_cells === 0 && (
                <p>No missing cells detected.</p>
              )}
              <small>Quality score considers missingness and duplicates.</small>
            </>
          )}
          {analysis.llm_insights && (
            <div className="notice">
              <b>Data-quality review</b>
              <p>{analysis.llm_insights.summary}</p>
              {(analysis.llm_insights.risks || [])
                .slice(0, 3)
                .map((item, index) => (
                  <p key={index}>{item}</p>
                ))}
              <small>
                Source: {analysis.llm_insights.source || "fallback"}
              </small>
            </div>
          )}
        </div>
        <div className="card">
          <h3>Data preview</h3>
          <div className="tablewrap">
            <table>
              <caption className="sr-only">First dataset rows</caption>
              <thead>
                <tr>
                  {columns.slice(0, 5).map((column) => (
                    <th key={column} scope="col">
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.slice(0, 6).map((row, index) => (
                  <tr key={index}>
                    {columns.slice(0, 5).map((column) => (
                      <td key={column} title={String(row[column] ?? "")}>
                        {String(row[column] ?? "N/A").slice(0, 120)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}
