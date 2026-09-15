import {mkdir, copyFile} from 'node:fs/promises';
await mkdir('static/vendor', {recursive: true});
for (const [source, target] of [
  ['alpinejs/dist/cdn.min.js', 'alpine.min.js'],
  ['htmx.org/dist/htmx.min.js', 'htmx.min.js'],
  ['plotly.js-dist-min/plotly.min.js', 'plotly.min.js'],
  ['htmx.org/LICENSE', 'htmx-LICENSE.txt'],
  ['plotly.js-dist-min/LICENSE', 'plotly-LICENSE.txt'],
  ['@vue/reactivity/LICENSE', 'vue-reactivity-LICENSE.txt'],
  ['@vue/shared/LICENSE', 'vue-shared-LICENSE.txt'],
  ['tailwindcss/LICENSE', 'tailwind-LICENSE.txt'],
]) await copyFile(`node_modules/${source}`, `static/vendor/${target}`);
