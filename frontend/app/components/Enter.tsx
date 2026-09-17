/**
 * Entrance choreography, Standard motion tier: 420 ms, 60 ms stagger.
 *
 * Deliberately CSS rather than framer-motion, which drives everything that
 * animates *state* on this page — the pipeline rail, the node fills, the edge
 * flows and the confidence bars. framer writes its `initial` into the
 * server-rendered HTML, so a stagger built that way ships `opacity: 0` to
 * anyone whose JavaScript is slow, blocked or broken. A keyframe with
 * `animation-fill-mode: backwards` keeps the resting state visible in the
 * markup and still plays the entrance, and it costs no JavaScript at all.
 *
 * There is no state and no effect here — the animation is entirely CSS. It
 * forwards a ref only so a wrapped section can be a scroll target.
 */

import { forwardRef, type ReactNode } from 'react';

type EnterProps = {
  children: ReactNode;
  index?: number;
  className?: string;
};

/**
 * Forwards its ref, because a section wrapped in one of these is sometimes
 * also a scroll target — the pipeline is, when a run starts.
 */
const Enter = forwardRef<HTMLDivElement, EnterProps>(function Enter(
  { children, index = 0, className },
  ref,
) {
  return (
    <div
      ref={ref}
      className={`enter${className ? ` ${className}` : ''}`}
      style={{ animationDelay: `${index * 60}ms` }}
    >
      {children}
    </div>
  );
});

export default Enter;
