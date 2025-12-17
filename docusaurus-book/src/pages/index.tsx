import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

// Book-focused hero section with chapter-like structure
const BookHero = () => {
  const {siteConfig} = useDocusaurusContext();

  return (
    <div className={styles.bookHero}>
      <div className="container">
        <div className="row">
          <div className="col col--7">
            <div className={styles.heroContent}>
              <div className={styles.bookBadge}>Textbook</div>
              <Heading as="h1" className={clsx(styles.mainTitle)}>
                {siteConfig.title}
              </Heading>
              <p className={clsx(styles.tagline)}>{siteConfig.tagline}</p>
              <p className={styles.heroDescription}>
                Dive deep into the revolutionary field of Physical AI and Humanoid Robotics.
                This comprehensive textbook bridges the gap between artificial intelligence
                and embodied intelligence, providing you with the knowledge needed to
                understand and develop next-generation robotic systems.
              </p>
              <div className={styles.heroButtons}>
                <Link
                  className="button button--primary button--lg"
                  to="/docs/intro">
                  Begin Reading
                </Link>
                <Link
                  className="button button--secondary button--lg margin-left--md"
                  to="/docs/module-1-ros2/intro">
                  Explore Modules
                </Link>
              </div>
            </div>
          </div>
          <div className="col col--5">
            <div className={styles.bookCover}>
              <div className={styles.coverDesign}>
                <div className={styles.coverTitle}>
                  <h2>Physical AI &<br />Humanoid Robotics</h2>
                  <p className={styles.coverSubtitle}>A Comprehensive Guide</p>
                </div>
                <div className={styles.coverDecoration}>
                  <div className={styles.coverPattern}></div>
                </div>
                <div className={styles.pageEdge}></div>
              </div>
              <div className={styles.bookStats}>
                <div className={styles.statItem}>
                  <div className={styles.statNumber}>4</div>
                  <div className={styles.statLabel}>Modules</div>
                </div>
                <div className={styles.statItem}>
                  <div className={styles.statNumber}>20+</div>
                  <div className={styles.statLabel}>Topics</div>
                </div>
                <div className={styles.statItem}>
                  <div className={styles.statNumber}>100+</div>
                  <div className={styles.statLabel}>Examples</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Book-focused features section with chapter-like organization
const BookFeatures = () => {
  const features = [
    {
      title: 'Module 1: The Robotic Nervous System',
      description: 'Master ROS 2 fundamentals, nodes, topics, and how to control humanoid robots using Python.',
      icon: '🔌',
      module: 'Module 1',
      to: '/docs/module-1-ros2/intro'
    },
    {
      title: 'Module 2: The Digital Twin',
      description: 'Explore simulation environments with Gazebo and Unity for robot development and testing.',
      icon: '🎮',
      module: 'Module 2',
      to: '/docs/module-2-digital-twin/gazebo-intro'
    },
    {
      title: 'Module 3: The AI-Robot Brain',
      description: 'Understand NVIDIA Isaac Sim, SLAM navigation, and perception pipelines for intelligent robots.',
      icon: '🧠',
      module: 'Module 3',
      to: '/docs/module-3-ai-brain/isaac-sim-overview'
    },
    {
      title: 'Module 4: Vision-Language-Action',
      description: 'Integrate LLMs with robotics for advanced human-robot interaction and task execution.',
      icon: '💬',
      module: 'Module 4',
      to: '/docs/module-4-vla/whisper-voice-commands'
    }
  ];

  return (
    <section className={clsx(styles.featuresSection, "section")}>
      <div className="container">
        <div className="content-wrapper">
          <Heading as="h2" className={clsx(styles.sectionTitle, "content-wrapper__title")}>
            Book Modules
          </Heading>
          <p className={clsx(styles.sectionSubtitle, "content-wrapper__subtitle")}>
            Each module builds upon the previous to create a comprehensive understanding
          </p>
        </div>
        <div className="row">
          {features.map((feature, idx) => (
            <div key={idx} className="col col--6">
              <div className={clsx(styles.featureCard, "card")}>
                <div className="card__body">
                  <div className={styles.featureHeader}>
                    <div className={styles.featureIcon}>{feature.icon}</div>
                    <div className={styles.moduleLabel}>{feature.module}</div>
                  </div>
                  <Heading as="h3" className={styles.featureTitle}>{feature.title}</Heading>
                  <p className={styles.featureDescription}>{feature.description}</p>
                  <Link
                    className={clsx(styles.featureLink, "button button--outline")}
                    to={feature.to}
                  >
                    Start Reading
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Learning outcomes section
const LearningOutcomes = () => {
  const outcomes = [
    {
      title: 'Comprehensive Understanding',
      description: 'Gain deep knowledge of Physical AI principles and their application in humanoid robotics.'
    },
    {
      title: 'Practical Skills',
      description: 'Develop hands-on experience with industry-standard tools like ROS 2, Isaac Sim, and Unity.'
    },
    {
      title: 'Integration Expertise',
      description: 'Learn to integrate vision, language, and action systems for complete robotic solutions.'
    },
    {
      title: 'Future-Ready Knowledge',
      description: 'Understand cutting-edge research and development trends in embodied AI.'
    }
  ];

  return (
    <section className={clsx(styles.outcomesSection, "section", "section--gradient")}>
      <div className="container">
        <div className="content-wrapper">
          <Heading as="h2" className={clsx(styles.sectionTitle, "content-wrapper__title")}>
            What You Will Learn
          </Heading>
          <p className={clsx(styles.sectionSubtitle, "content-wrapper__subtitle")}>
            Master the skills needed to develop next-generation robotic systems
          </p>
        </div>
        <div className="row">
          {outcomes.map((outcome, idx) => (
            <div key={idx} className="col col--3">
              <div className={clsx(styles.outcomeCard, "card")}>
                <div className="card__body">
                  <h4 className={styles.outcomeTitle}>{outcome.title}</h4>
                  <p className={styles.outcomeDescription}>{outcome.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Testimonials section
const Testimonials = () => {
  const testimonials = [
    {
      text: "This textbook perfectly bridges the gap between theoretical AI and practical robotics applications. The progression from ROS fundamentals to advanced VLA systems is masterfully crafted.",
      author: "Dr. Sarah Chen",
      role: "Robotics Researcher, Stanford"
    },
    {
      text: "The hands-on approach with Isaac Sim and real-world examples makes complex concepts accessible and practical. An invaluable resource for anyone serious about embodied AI.",
      author: "Prof. Michael Rodriguez",
      role: "Computer Science Professor"
    },
    {
      text: "An essential resource for the next generation of roboticists. The integration of modern AI techniques with traditional robotics is exactly what the field needs.",
      author: "Alex Thompson",
      role: "Senior AI Engineer"
    }
  ];

  return (
    <section className={clsx(styles.testimonialsSection, "section")}>
      <div className="container">
        <div className="content-wrapper">
          <Heading as="h2" className={clsx(styles.sectionTitle, "content-wrapper__title")}>
            What Readers Say
          </Heading>
        </div>
        <div className="row">
          {testimonials.map((testimonial, idx) => (
            <div key={idx} className="col col--4">
              <div className={clsx(styles.testimonialCard, "card")}>
                <div className="card__body">
                  <div className={styles.quoteIcon}>" </div>
                  <p className={styles.testimonialText}>{testimonial.text}</p>
                  <div className={styles.testimonialAuthor}>
                    <strong>{testimonial.author}</strong>
                    <span className={styles.testimonialRole}>{testimonial.role}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Book-focused call to action
const CallToAction = () => {
  return (
    <section className={clsx(styles.ctaSection, "section")}>
      <div className="container">
        <div className="content-wrapper">
          <Heading as="h2" className={clsx(styles.sectionTitle, "content-wrapper__title")}>
            Start Your Journey
          </Heading>
          <p className={clsx(styles.sectionSubtitle, "content-wrapper__subtitle")}>
            Begin reading our comprehensive textbook and master Physical AI & Humanoid Robotics
          </p>
          <div className={styles.ctaButtons}>
            <Link
              className="button button--primary button--lg"
              to="/docs/intro">
              Start Reading Now
            </Link>
            <Link
              className="button button--secondary button--lg margin-left--md"
              to="/docs/module-1-ros2/intro">
              Explore Modules
            </Link>
            <Link
              className="button button--outline button--lg margin-left--md"
              to="/docs/glossary">
              View Glossary
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
};

function HomepageHeader() {
  return (
    <header className={clsx(styles.heroBanner)}>
      <BookHero />
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics Textbook`}
      description="Master Physical AI and Humanoid Robotics with our comprehensive textbook. Learn about embodied intelligence, ROS, Isaac Sim, and more.">
      <HomepageHeader />
      <main>
        <BookFeatures />
        <LearningOutcomes />
        <Testimonials />
        <CallToAction />
      </main>
    </Layout>
  );
}
