import React, { useEffect } from 'react'
import { FeedbackWidget } from '@/components/ui/feedback-widget'

export default function FeedbackModal({ isOpen, onClose, onSubmitFeedback, initialRating = "neutral" }) {
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') onClose?.()
    }
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown)
      return () => window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div
      className="feedback-modal-backdrop"
      role="presentation"
      onMouseDown={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        onMouseDown={(e) => e.stopPropagation()}
        className="feedback-modal-wrapper"
      >
        <FeedbackWidget
          initialRating={initialRating}
          onSubmit={async (data) => {
            await onSubmitFeedback?.(data)
            onClose?.()
          }}
          onClose={onClose}
        />
      </div>
    </div>
  )
}


