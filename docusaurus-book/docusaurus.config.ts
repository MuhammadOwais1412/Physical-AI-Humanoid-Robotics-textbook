import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'Bridging Digital AI and Embodied Intelligence',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://physical-ai-humanoid-robotics-textbook.vercel.app', // Update with your actual Vercel deployment URL
  // Set the /<baseUrl>/ pathname under which your site is served
  // For Vercel deployment, use '/' for root domain or '/repo-name' if in subdirectory
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'your-organization', // Update with actual GitHub organization/user name
  projectName: 'Physical-AI-Humanoid-Robotics-textbook', // Usually your repo name.

  onBrokenLinks: 'throw',
  markdown: {
    format: 'mdx',
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: 'https://github.com/your-org/Physical-AI-Humanoid-Robotics-textbook/edit/main/', // Update with actual repository URL
          // Options for docs versioning
          showLastUpdateAuthor: false,
          showLastUpdateTime: false,
        },
        theme: {
          customCss: './src/css/book-theme.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    // Accessibility metadata
    metadata: [
      {
        name: 'keywords',
        content: 'robotics, AI, humanoid, ROS, physical AI, machine learning, computer vision, natural language processing'
      },
      {
        name: 'author',
        content: 'Physical AI & Humanoid Robotics Textbook Team'
      },
      {
        name: 'og:locale',
        content: 'en_US'
      },
      {
        name: 'og:site_name',
        content: 'Physical AI & Humanoid Robotics Textbook'
      },
      // WCAG 2.1 AA compliance indicators
      {
        name: 'Accessibility',
        content: 'WCAG 2.1 AA compliant'
      }
    ],
    // algolia: {
    //   // The application ID provided by Algolia
    //   appId: 'YOUR_ALGOLIA_APP_ID',
    //   // Public API key: it is safe to commit it
    //   apiKey: 'YOUR_ALGOLIA_API_KEY',
    //   indexName: 'physical-ai-textbook',
    //   // Optional: see doc section below
    //   contextualSearch: true,
    //   // Optional: Specify domains where the navigation should occur through window.location instead on history.push. Useful when our Algolia config crawls multiple documentation sites and we want to navigate with window.location.href to them.
    //   externalUrlRegex: 'external\\.com|domain\\.com',
    //   // Optional: Replace parts of the item URLs from Algolia. Useful when using the same search index for multiple deployments using a different baseUrl. You can use regexp or string in the `from` param. For example: localhost:3000 vs myCompany.com/docs
    //   replaceSearchResultPathname: {
    //     from: '/docs/', // or as RegExp: /\/docs\//
    //     to: '/',
    //   },
    //   // Optional: Algolia search parameters
    //   searchParameters: {},
    //   // Optional: path for search page that enabled by default (`false` to disable it)
    //   searchPagePath: 'search',
    // },
    navbar: {
      title: 'Physical AI Textbook',
      logo: {
        alt: 'Physical AI & Humanoid Robotics Logo',
        src: 'img/logo.png',
        srcDark: 'img/logo.png', // Add dark mode logo if available
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'textbook',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/your-org/Physical-AI-Humanoid-Robotics-textbook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'Module 1: ROS 2',
              to: '/docs/module-1-ros2/intro',
            },
            {
              label: 'Module 2: Digital Twin',
              to: '/docs/module-2-digital-twin/gazebo-intro',
            },
            {
              label: 'Module 3: AI-Robot Brain',
              to: '/docs/module-3-ai-brain/isaac-sim-overview',
            },
            {
              label: 'Module 4: Vision-Language-Action',
              to: '/docs/module-4-vla/whisper-voice-commands',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Glossary',
              to: '/docs/glossary',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/your-org/Physical-AI-Humanoid-Robotics-textbook',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
