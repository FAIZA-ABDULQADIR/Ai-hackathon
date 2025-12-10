// Example plugin configuration for Docusaurus
// This demonstrates how to add custom plugins to extend functionality

// Adding a custom plugin for search enhancement
plugins: [
  [
    '@docusaurus/plugin-content-docs',
    {
      id: 'community',
      path: 'community',
      routeBasePath: 'community',
      sidebarPath: require.resolve('./sidebarsCommunity.js'),
    },
  ],
  // Adding sitemap plugin for SEO
  [
    '@docusaurus/plugin-sitemap',
    {
      changefreq: 'weekly',
      priority: 0.5,
      filename: 'sitemap.xml',
    },
  ],
  // Adding client modules for custom CSS
  [
    '@docusaurus/plugin-client-redirects',
    {
      redirects: [
        {
          to: '/docs/new-doc', // new url
          from: ['/docs/old-doc', '/docs/legacy-doc'], // old urls
        },
      ],
    },
  ],
],

// Example of using remark and rehype plugins for MDX processing
presets: [
  [
    'classic',
    {
      docs: {
        sidebarPath: require.resolve('./sidebars.js'),
        // Add custom remark/rehype plugins
        remarkPlugins: [
          [require('@docusaurus/remark-plugin-npm2yarn'), { sync: true }],
        ],
      },
      theme: {
        customCss: require.resolve('./src/css/custom.css'),
      },
    },
  ],
],

// Example of using a third-party plugin
// First install: npm install --save-dev @docusaurus/plugin-content-pages
plugins: [
  [
    '@docusaurus/plugin-content-pages',
    {
      path: 'src/pages',
      routeBasePath: '/',
      include: ['**/*.{js,jsx,ts,tsx,md,mdx}'],
    },
  ],
],