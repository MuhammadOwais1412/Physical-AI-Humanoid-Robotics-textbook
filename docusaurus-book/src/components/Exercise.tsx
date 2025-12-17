import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './Exercise.module.css';

interface ExerciseProps {
  id?: string;
  title: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  type: 'practical' | 'theoretical' | 'coding' | 'research';
  instructions: string;
  hint?: string;
  solution?: string;
  children?: React.ReactNode;
}

const difficultyColors = {
  beginner: 'success',
  intermediate: 'warning',
  advanced: 'danger',
};

const typeLabels = {
  practical: 'Practical',
  theoretical: 'Theoretical',
  coding: 'Coding',
  research: 'Research',
};

const Exercise: React.FC<ExerciseProps> = ({
  id,
  title,
  difficulty,
  type,
  instructions,
  hint,
  solution,
  children,
}) => {
  const [showHint, setShowHint] = useState(false);
  const [showSolution, setShowSolution] = useState(false);

  return (
    <div className={clsx('card', styles.exerciseCard)}>
      <div className="card__header">
        <h3 className={styles.exerciseTitle}>
          {title}
          <span className={`badge badge--${difficultyColors[difficulty]}`} style={{ marginLeft: '0.5rem' }}>
            {difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}
          </span>
          <span className="badge badge--primary" style={{ marginLeft: '0.5rem' }}>
            {typeLabels[type]}
          </span>
        </h3>
      </div>
      <div className="card__body">
        <div className={styles.instructions}>
          <h4>Instructions</h4>
          <p>{instructions}</p>
          {children && <div className={styles.exerciseContent}>{children}</div>}
        </div>

        {hint && (
          <div className={styles.hintSection}>
            <button
              className="button button--secondary button--sm"
              onClick={() => setShowHint(!showHint)}
            >
              {showHint ? 'Hide Hint' : 'Show Hint'}
            </button>
            {showHint && (
              <div className={clsx('alert alert--info', styles.hint)}>
                <p>{hint}</p>
              </div>
            )}
          </div>
        )}

        {solution && (
          <div className={styles.solutionSection}>
            <button
              className="button button--primary button--sm"
              onClick={() => setShowSolution(!showSolution)}
            >
              {showSolution ? 'Hide Solution' : 'Show Solution'}
            </button>
            {showSolution && (
              <div className={clsx('alert alert--success', styles.solution)}>
                <h5>Solution:</h5>
                <div>{solution}</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Exercise;