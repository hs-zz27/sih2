# Production image for the Next.js frontend.
#
# This ran `npm run dev` - the development server - as its container command,
# behind `npm install` rather than a lockfile-faithful install. That is not a
# packaging nicety: `next dev` compiles routes on demand, ships unminified
# bundles with hot-reload machinery attached, and is explicitly not supported
# for serving. On demo day it is also the slowest possible first paint.
#
# Two stages: build once, then run the built output.

FROM node:20-slim AS builder

WORKDIR /app

# `npm ci` fails loudly if package-lock.json disagrees with package.json,
# which is the point: an image whose dependency versions drift from the
# lockfile is not the thing that was tested.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Next.js inlines NEXT_PUBLIC_* into the client bundle at BUILD time. Passing
# it at run time through `environment:` does nothing to an already-compiled
# bundle, so it arrives here as a build argument (see docker-compose.yml).
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build


FROM node:20-slim AS runner

WORKDIR /app
ENV NODE_ENV=production

# node:20-slim ships a `node` user (uid 1000). Running as root inside a
# container that serves HTTP buys nothing and costs the usual.
COPY --from=builder --chown=node:node /app/package.json /app/package-lock.json ./
COPY --from=builder --chown=node:node /app/node_modules ./node_modules
COPY --from=builder --chown=node:node /app/.next ./.next

USER node

EXPOSE 3000

CMD ["npm", "run", "start"]
