import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";

const markdownComponents: Components = {
  a: ({ node: _n, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer" />
  ),
};

type Props = {
  markdown: string;
  className?: string;
  /** When true, show a short empty placeholder */
  emptyLabel?: string;
  /** Render nothing when markdown is empty */
  hideWhenEmpty?: boolean;
};

export function MarkdownPreview({
  markdown,
  className = "",
  emptyLabel = "(empty)",
  hideWhenEmpty = false,
}: Props) {
  const trimmed = (markdown || "").trim();
  if (!trimmed) {
    if (hideWhenEmpty) {
      return null;
    }
    return <p className={`markdown-empty muted-note${className ? ` ${className}` : ""}`}>{emptyLabel}</p>;
  }
  return (
    <div className={`markdown-body${className ? ` ${className}` : ""}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
