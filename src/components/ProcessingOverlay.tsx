import { useProjectStore } from '../hooks/store';

export default function ProcessingOverlay() {
  const { isProcessing, progress, progressMessage, error, setError } = useProjectStore();

  if (!isProcessing && !error) return null;

  return (
    <div className="processing-overlay">
      <div className="processing-card">
        {isProcessing ? (
          <>
            <div className="processing-spinner" />
            <h3 className="processing-title">Processing</h3>
            <p className="processing-message">{progressMessage || 'Please wait...'} </p>
            <div className="progress-bar">
              <div
                className="progress-bar-fill"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <span className="progress-text">{Math.round(progress * 100)}%</span>
          </>
        ) : error ? (
          <>
            <div className="error-icon">!</div>
            <h3 className="processing-title">Error</h3>
            <p className="error-message">{error}</p>
            <button className="btn btn-danger" onClick={() => setError(null)}>
              Dismiss
            </button>
          </>
        ) : null}
      </div>
    </div>
  );
}
