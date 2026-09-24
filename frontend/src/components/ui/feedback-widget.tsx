"use client";

import { cn } from "@/lib/utils";
import * as ToggleGroup from "@radix-ui/react-toggle-group";
import { AnimatePresence, motion } from "motion/react";
import * as React from "react";
import ReactMarkdown from "react-markdown";

const EMOJIS = [
  {
    id: "very-sad",
    label: "Terrible",
    icon: (
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M16 16s-1.5-2-4-2-4 2-4 2" />
        <path d="M9 9h.01" />
        <path d="M15 9h.01" />
        <path d="M9 13v2" stroke="#3b82f6" />
        <path d="M15 13v2" stroke="#3b82f6" />
      </svg>
    ),
  },
  {
    id: "sad",
    label: "Bad",
    icon: (
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M16 16s-1.5-2-4-2-4 2-4 2" />
        <line x1="9" y1="9" x2="9.01" y2="9" />
        <line x1="15" y1="9" x2="15.01" y2="9" />
      </svg>
    ),
  },
  {
    id: "neutral",
    label: "Okay",
    icon: (
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M8 13s1.5 2 4 2 4-2 4-2" />
        <line x1="9" y1="9" x2="9.01" y2="9" />
        <line x1="15" y1="9" x2="15.01" y2="9" />
      </svg>
    ),
  },
  {
    id: "happy",
    label: "Amazing",
    icon: (
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M8 13s1.5 2 4 2 4-2 4-2" />
        <path
          d="M9 9l.5 1.5l1.5 .5l-1.5 .5l-.5 1.5l-.5-1.5l-1.5-.5l1.5-.5z"
          fill="#f97316"
          stroke="none"
        />
        <path
          d="M15 9l.5 1.5l1.5 .5l-1.5 .5l-.5 1.5l-.5-1.5l-1.5-.5l1.5-.5z"
          fill="#f97316"
          stroke="none"
        />
      </svg>
    ),
  },
];

interface FeedbackWidgetProps {
  onSubmit?: (data: { rating: string; feedback: string }) => void;
  onClose?: () => void;
  className?: string;
  /** Text shown in the collapsed state */
  label?: string;
  /** Placeholder for the textarea */
  placeholder?: string;
  /** Initial rating ID to pre-expand the widget */
  initialRating?: string;
}

export function FeedbackWidget({
  onSubmit,
  onClose,
  className,
  label = "Was this helpful?",
  placeholder = "Your feedback...",
  initialRating = "",
}: FeedbackWidgetProps) {
  const [value, setValue] = React.useState<string>(initialRating);
  const [feedback, setFeedback] = React.useState("");
  const [isPreview, setIsPreview] = React.useState(false);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const isExpanded = value !== "";
  const containerRef = React.useRef<HTMLDivElement>(null);

  const springTransition = {
    type: "spring",
    stiffness: 300,
    damping: 30,
    mass: 1,
  } as const;

  const handleValueChange = (val: string) => {
    if (val === "" || val === value) {
      setValue("");
      setIsPreview(false);
      onClose?.();
    } else {
      setValue(val);
      // Auto-focus the textarea after expansion
      setTimeout(() => {
        containerRef.current?.querySelector("textarea")?.focus();
      }, 100);
    }
  };

  const handleSend = async () => {
    if (!feedback.trim()) return;

    setIsSubmitting(true);
    try {
      await onSubmit?.({ rating: value, feedback });
      setValue("");
      setFeedback("");
      setIsPreview(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={cn("feedback-modal-wrapper", className)}>
      <motion.div
        ref={containerRef}
        layout
        transition={springTransition}
        initial={false}
        className={cn(
          "feedback-widget-card",
          isExpanded ? "feedback-card-expanded" : "feedback-card-collapsed"
        )}
      >
        <motion.div layout="position" transition={springTransition}>
          <div className="feedback-widget-header">
            <span className="feedback-widget-title">
              {label}
            </span>

            <ToggleGroup.Root
              type="single"
              value={value}
              onValueChange={handleValueChange}
              className="feedback-emoji-group"
            >
              {EMOJIS.map((emoji) => (
                <ToggleGroup.Item key={emoji.id} value={emoji.id} asChild>
                  <button
                    type="button"
                    title={emoji.label}
                    className={cn(
                      "feedback-emoji-btn",
                      value === emoji.id && "active"
                    )}
                  >
                    <motion.div
                      layout="position"
                      transition={springTransition}
                      className="relative z-10 flex h-5 w-5 scale-110 items-center justify-center transition-transform active:scale-90"
                    >
                      {emoji.icon}
                    </motion.div>
                    {value === emoji.id && (
                      <motion.div
                        layoutId="active-bg"
                        className="absolute inset-0 rounded-full bg-blue-600"
                        transition={springTransition}
                      />
                    )}
                  </button>
                </ToggleGroup.Item>
              ))}
            </ToggleGroup.Root>
          </div>

          <AnimatePresence mode="popLayout" initial={false}>
            {isExpanded && (
              <motion.div
                initial={{
                  height: 0,
                  opacity: 0,
                  scale: 0.98,
                  filter: "blur(4px)",
                }}
                animate={{
                  height: "auto",
                  opacity: 1,
                  scale: 1,
                  filter: "blur(0px)",
                }}
                exit={{
                  height: 0,
                  opacity: 0,
                  scale: 0.98,
                  filter: "blur(4px)",
                  transition: {
                    height: { duration: 0.3, ease: [0.32, 0, 0.67, 0] },
                    opacity: { duration: 0.15 },
                    scale: { duration: 0.2 },
                    filter: { duration: 0.2 },
                  },
                }}
                transition={{
                  height: { ...springTransition, bounce: 0 },
                  opacity: { duration: 0.25 },
                  scale: { ...springTransition, damping: 25 },
                  filter: { duration: 0.3 },
                }}
                className="overflow-hidden"
              >
                <div>
                  <div className="feedback-section-header">
                    <span className="feedback-section-label">
                      {isPreview ? "Preview" : "Feedback"}
                    </span>
                    <button
                      type="button"
                      onClick={() => setIsPreview(!isPreview)}
                      className="feedback-preview-btn"
                    >
                      {isPreview ? "Edit" : "Preview"}
                    </button>
                  </div>

                  <div className="feedback-textarea-wrapper">
                    <AnimatePresence mode="wait">
                      {isPreview ? (
                        <motion.div
                          key="preview"
                          initial={{ opacity: 0, y: 5 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -5 }}
                          className="feedback-preview-content prose prose-invert"
                        >
                          <ReactMarkdown>
                            {feedback || "*Nothing to preview...*"}
                          </ReactMarkdown>
                        </motion.div>
                      ) : (
                        <motion.textarea
                          key="editor"
                          initial={{ opacity: 0, y: 5 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -5 }}
                          autoFocus
                          placeholder={placeholder}
                          value={feedback}
                          onChange={(e) => setFeedback(e.target.value)}
                          className="feedback-textarea"
                        />
                      )}
                    </AnimatePresence>

                    {!isPreview && (
                      <div className="feedback-watermark">
                        <span>M↓</span>
                        <span>supported</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="feedback-divider-line" />

                <motion.div
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  exit={{ y: 20, opacity: 0 }}
                  transition={{ delay: 0.1, ...springTransition }}
                  className="feedback-footer-row"
                >
                  <p className="feedback-footer-note">
                    We appreciate your input.
                  </p>
                  <button
                    type="button"
                    onClick={handleSend}
                    disabled={!feedback.trim() || isSubmitting}
                    className="feedback-send-btn"
                  >
                    {isSubmitting ? (
                      <span className="feedback-spinner" />
                    ) : (
                      "Send Feedback"
                    )}
                  </button>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>
    </div>
  );
}

export default FeedbackWidget;


