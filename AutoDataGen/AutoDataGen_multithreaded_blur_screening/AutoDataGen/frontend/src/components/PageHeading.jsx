import { ArrowLeft } from "lucide-react";

export default function PageHeading({ type, children }) {
  return (
    <section className="hero workspace-heading">
      <a className="back-link" href="#/">
        <ArrowLeft size={16} aria-hidden="true" /> Change dataset type
      </a>
      <small>DATASET AUGMENTATION / {type.toUpperCase()}</small>
      <h1>{type} augmentation</h1>
      <p>{children}</p>
    </section>
  );
}
