import './style.css';
import './guide.css';

// Highlight the section you're reading in the table of contents.
const links = new Map([...document.querySelectorAll('.toc a')].map((a) => [a.getAttribute('href').slice(1), a]));
const observer = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    links.forEach((a) => a.removeAttribute('aria-current'));
    links.get(e.target.id)?.setAttribute('aria-current', 'true');
  }
}, { rootMargin: '-20% 0px -70% 0px' });
document.querySelectorAll('.guide-body section[id]').forEach((s) => observer.observe(s));


if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch(() => {});
}
