export default function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <b>{value ?? "N/A"}</b>
    </div>
  );
}
