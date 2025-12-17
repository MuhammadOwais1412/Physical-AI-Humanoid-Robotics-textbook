import React, { useState } from 'react';
import Mermaid from 'react-mermaid';
import clsx from 'clsx';
import styles from './Diagram.module.css';

interface DiagramProps {
  id?: string;
  title: string;
  description?: string;
  children?: React.ReactNode;
  type?: 'static' | 'interactive' | 'simulation' | 'mermaid';
  width?: string;
  height?: string;
  mermaidCode?: string;
  mermaidType?: 'flowchart' | 'sequence' | 'gantt' | 'class' | 'state' | 'pie' | 'er' | 'user-journey';
}

const Diagram: React.FC<DiagramProps> = ({
  id,
  title,
  description,
  children,
  type = 'static',
  width = '100%',
  height = 'auto',
  mermaidCode,
  mermaidType = 'flowchart',
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Handle mermaid diagram rendering
  const renderMermaid = () => {
    if (!mermaidCode) return null;

    // Set the diagram type in the Mermaid code if not specified
    let diagramCode = mermaidCode.trim();
    if (!diagramCode.startsWith(`${mermaidType.toLowerCase()}`)) {
      diagramCode = `${mermaidType.toLowerCase()} TD\n${diagramCode}`;
    }

    return <Mermaid chart={diagramCode} className={styles.mermaid} />;
  };

  return (
    <div className={clsx('card', styles.diagramCard)} style={{ width }}>
      <div className="card__header">
        <h3 className={styles.diagramTitle}>
          {title}
          {type !== 'static' && (
            <span className="badge badge--secondary" style={{ marginLeft: '0.5rem' }}>
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </span>
          )}
        </h3>
      </div>
      <div className="card__body" style={{ height }}>
        <div className={styles.diagramContainer}>
          {type === 'mermaid' ? renderMermaid() : children}
        </div>

        {description && (
          <div className={styles.description}>
            <p>{description}</p>
          </div>
        )}

        {(type === 'interactive' || type === 'simulation') && (
          <div className={styles.controls}>
            <button
              className="button button--primary button--sm"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? 'Collapse' : 'Expand'} View
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Diagram;