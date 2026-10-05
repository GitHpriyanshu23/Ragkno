import { Check, FileText, X } from 'lucide-react'

export default function UploadProgress({ upload, onDismiss }) {
  if (!upload) return null
  const active = upload.stage === 'uploading' || upload.stage === 'indexing'
  const status = upload.stage === 'indexing' ? 'Indexing documents…' : upload.stage === 'uploading' ? 'Uploading…' : upload.stage === 'error' ? 'Upload failed' : 'Ready to ask questions'
  return (
    <section className={`upload-progress-card is-${upload.stage}`} aria-label="Document upload progress">
      <div className="upload-progress-icon"><FileText size={27} /></div>
      <div className="upload-progress-body">
        <strong title={upload.name}>{upload.name}</strong>
        <p>{upload.size < 1048576 ? `${(upload.size / 1024).toFixed(1)} KB` : `${(upload.size / 1048576).toFixed(1)} MB`} · <span role="status">{status}</span></p>
        <div className="upload-progress-bottom">
          <div className="upload-progress-track" role="progressbar" aria-label={upload.stage === 'indexing' ? 'Indexing documents' : 'Uploading documents'} aria-valuemin={0} aria-valuemax={100} aria-valuenow={upload.stage === 'indexing' ? undefined : upload.percent} aria-valuetext={status}>
            <div className="upload-progress-fill" style={{ width: `${upload.percent}%` }} />
          </div>
          <span className="upload-progress-value">{upload.stage === 'indexing' ? 'Indexing' : upload.stage === 'done' ? <Check size={18} aria-label="Complete" /> : upload.stage === 'error' ? 'Failed' : `${upload.percent}%`}</span>
        </div>
        {upload.error && <p className="upload-progress-error">{upload.error}</p>}
      </div>
      {!active && <button type="button" className="upload-progress-dismiss" onClick={onDismiss} aria-label="Dismiss upload progress"><X size={22} /></button>}
    </section>
  )
}
