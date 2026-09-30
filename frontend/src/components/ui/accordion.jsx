import React, { createContext, useContext, useState } from "react";
import { motion, AnimatePresence } from "motion/react";

const AccordionContext = createContext(null);
const AccordionItemContext = createContext(null);

export function Accordion({
  children,
  type = "single",
  collapsible = true,
  defaultValue = null,
  value,
  onValueChange,
  className = "",
}) {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const activeValue = value !== undefined ? value : internalValue;

  const handleToggle = (itemValue) => {
    let nextValue;
    if (type === "single") {
      if (activeValue === itemValue) {
        nextValue = collapsible ? null : activeValue;
      } else {
        nextValue = itemValue;
      }
    } else {
      const currentList = Array.isArray(activeValue) ? activeValue : [];
      if (currentList.includes(itemValue)) {
        nextValue = currentList.filter((v) => v !== itemValue);
      } else {
        nextValue = [...currentList, itemValue];
      }
    }

    if (value === undefined) {
      setInternalValue(nextValue);
    }
    onValueChange?.(nextValue);
  };

  const isItemOpen = (itemValue) => {
    if (type === "single") {
      return activeValue === itemValue;
    }
    return Array.isArray(activeValue) && activeValue.includes(itemValue);
  };

  return (
    <AccordionContext.Provider value={{ handleToggle, isItemOpen }}>
      <div className={`faq-accordion-list ${className}`.trim()}>{children}</div>
    </AccordionContext.Provider>
  );
}

export function AccordionItem({ value, children, className = "" }) {
  const context = useContext(AccordionContext);
  if (!context) {
    throw new Error("AccordionItem must be used within Accordion");
  }

  const isOpen = context.isItemOpen(value);

  return (
    <AccordionItemContext.Provider value={{ value, isOpen }}>
      <div className={`faq-accordion-item ${className}`.trim()}>
        {children}
      </div>
    </AccordionItemContext.Provider>
  );
}

export function AccordionTrigger({ children, className = "" }) {
  const accordionContext = useContext(AccordionContext);
  const itemContext = useContext(AccordionItemContext);

  if (!accordionContext || !itemContext) {
    throw new Error("AccordionTrigger must be used within AccordionItem");
  }

  const { handleToggle } = accordionContext;
  const { value, isOpen } = itemContext;

  return (
    <button
      type="button"
      onClick={() => handleToggle(value)}
      aria-expanded={isOpen}
      className={`faq-trigger-btn ${className}`.trim()}
    >
      <span className="faq-question-text">{children}</span>
      <div className={`faq-icon-box ${isOpen ? "is-open" : ""}`.trim()}>
        <motion.span
          className="faq-icon-symbol"
          animate={{ rotate: isOpen ? 45 : 0 }}
          transition={{ duration: 0.22, ease: "easeInOut" }}
        >
          +
        </motion.span>
      </div>
    </button>
  );
}

export function AccordionContent({ children, className = "" }) {
  const itemContext = useContext(AccordionItemContext);
  if (!itemContext) {
    throw new Error("AccordionContent must be used within AccordionItem");
  }

  const { isOpen } = itemContext;

  return (
    <AnimatePresence initial={false}>
      {isOpen && (
        <motion.div
          key="content"
          initial={{ height: 0, opacity: 0 }}
          animate={{
            height: "auto",
            opacity: 1,
            transition: {
              height: { duration: 0.32, ease: [0.16, 1, 0.3, 1] },
              opacity: { duration: 0.2, delay: 0.05 },
            },
          }}
          exit={{
            height: 0,
            opacity: 0,
            transition: {
              height: { duration: 0.22, ease: [0.16, 1, 0.3, 1] },
              opacity: { duration: 0.12 },
            },
          }}
          className="faq-content-box"
        >
          <div className={`faq-answer-inner ${className}`.trim()}>
            {children}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
