import { beforeEach, expect, it, vi } from 'vite-plus/test'
import { checkFirecrawlTaskStatus } from './datasets'

const api = vi.hoisted(() => ({
  getCrawlStatus: vi.fn(),
}))

vi.mock('./console', () => ({
  consoleClient: {
    website: {
      crawl: {
        status: {
          byJobId: {
            get: api.getCrawlStatus,
          },
        },
      },
    },
  },
}))

beforeEach(() => {
  api.getCrawlStatus.mockReset()
})

it('keeps distinct server handles in status requests that share a provider job id', () => {
  checkFirecrawlTaskStatus('shared-job', 'source-a')
  checkFirecrawlTaskStatus('shared-job', 'source-b')

  expect(api.getCrawlStatus).toHaveBeenNthCalledWith(
    1,
    {
      params: { job_id: 'shared-job' },
      query: { provider: 'firecrawl', source_handle: 'source-a' },
    },
    { context: { silent: true } },
  )
  expect(api.getCrawlStatus).toHaveBeenNthCalledWith(
    2,
    {
      params: { job_id: 'shared-job' },
      query: { provider: 'firecrawl', source_handle: 'source-b' },
    },
    { context: { silent: true } },
  )
})
