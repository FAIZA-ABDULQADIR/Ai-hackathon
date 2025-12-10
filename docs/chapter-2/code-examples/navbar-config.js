// Example navbar configuration for Docusaurus
// This shows how to configure the navigation bar in docusaurus.config.js

themeConfig: {
  navbar: {
    title: 'My Technical Book',
    logo: {
      alt: 'Logo',
      src: 'img/logo.svg',
    },
    items: [
      // Documentation sidebar link
      {
        type: 'docSidebar',
        sidebarId: 'tutorialSidebar',
        position: 'left',
        label: 'Book',
      },
      // Blog link
      { to: '/blog', label: 'Blog', position: 'left' },
      // Custom page link
      {
        href: 'https://github.com/your-username/my-technical-book',
        label: 'GitHub',
        position: 'right',
      },
    ],
  },
},

// Alternative navbar configuration with dropdown menu
themeConfig: {
  navbar: {
    title: 'My Technical Book',
    logo: {
      alt: 'Logo',
      src: 'img/logo.svg',
    },
    items: [
      {
        type: 'docSidebar',
        sidebarId: 'tutorialSidebar',
        position: 'left',
        label: 'Documentation',
      },
      {
        type: 'dropdown',
        label: 'More',
        position: 'left',
        items: [
          {
            label: 'Blog',
            to: '/blog',
          },
          {
            label: 'GitHub',
            href: 'https://github.com/your-username/my-technical-book',
          },
        ],
      },
    ],
  },
},