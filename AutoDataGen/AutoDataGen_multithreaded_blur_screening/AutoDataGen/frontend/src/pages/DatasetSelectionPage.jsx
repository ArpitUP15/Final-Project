import { ArrowRight, FileText, Image, Table2 } from "lucide-react";
import { DATASET_TYPES } from "../lib/datasetTypes.js";

const icons = { text: FileText, table: Table2, image: Image };

export default function DatasetSelectionPage() {
  return (
    <>
      <section className="hero selection-heading">
        <small>AUTODATAGEN WORKSPACE</small>
        <h1>
          What would you like
          <br />
          to augment?
        </h1>
        <p>
          Choose your dataset type to open its dedicated workspace. Upload,
          analyze, and augment your data from there.
        </p>
      </section>
      <section className="dataset-options" aria-label="Choose a dataset type">
        {DATASET_TYPES.map((type, index) => {
          const Icon = icons[type.id];
          return (
            <a
              key={type.id}
              href={type.href}
              className="dataset-option"
              aria-label={`Open ${type.label.toLowerCase()} augmentation`}
            >
              <div className="option-top">
                <Icon size={28} aria-hidden="true" />
                <span>0{index + 1}</span>
              </div>
              <h2>{type.label}</h2>
              <p>{type.description}</p>
              <ul>
                {type.details.map((detail) => (
                  <li key={detail}>{detail}</li>
                ))}
              </ul>
              <div className="option-formats">{type.formats}</div>
              <div className="option-action">
                Open {type.label.toLowerCase()} workspace{" "}
                <ArrowRight size={18} aria-hidden="true" />
              </div>
            </a>
          );
        })}
      </section>
      <p className="selection-note">
        Your selected workspace shows only the upload options, settings, and
        results for that dataset type.
      </p>
    </>
  );
}
