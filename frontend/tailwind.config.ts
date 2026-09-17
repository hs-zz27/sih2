import type { Config } from 'tailwindcss';

/**
 * The approved deck tokens, in one place.
 *
 * Every value here is also emitted as a CSS custom property in globals.css,
 * because the component layer and the OpenLayers/three.js code both need to
 * reach the same colours without importing a JS module. Tailwind reads the
 * variables rather than duplicating the hexes, so there is exactly one
 * definition of "champagne" in the codebase.
 */
const config: Config = {
  content: ['./app/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        void: 'var(--void)',
        sunk: 'var(--sunk)',
        pane: 'var(--pane)',
        raised: 'var(--raised)',
        hair: 'var(--hair)',
        'hair-lit': 'var(--hair-lit)',
        ink: 'var(--ink)',
        muted: 'var(--muted)',
        dim: 'var(--dim)',
        champ: 'var(--champ)',
        bronze: 'var(--bronze)',
        sage: 'var(--sage)',
        amber: 'var(--amber)',
        clay: 'var(--clay)',
      },
      fontFamily: {
        display: 'var(--f-display)',
        ui: 'var(--f-ui)',
        mono: 'var(--f-mono)',
      },
      // Density dial 8/10: a dashboard scale, not a marketing page.
      spacing: {
        s1: '8px',
        s2: '12px',
        s3: '16px',
        s4: '24px',
        s5: '32px',
      },
      borderRadius: {
        deck: '3px',
        chip: '2px',
      },
      maxWidth: {
        shell: '1400px',
        prose: '62ch',
      },
    },
  },
  plugins: [],
};

export default config;
