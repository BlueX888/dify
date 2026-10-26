import type {
  ExternalHitTestingPayload,
  HitTestingPayload as ConsoleHitTestingPayload,
} from './generated/api/console/datasets/types.gen.ts'
import {
  zExternalHitTestingPayload,
  zHitTestingPayload as zConsoleHitTestingPayload,
} from './generated/api/console/datasets/zod.gen.ts'
import type { HitTestingPayload as ServiceHitTestingPayload } from './generated/api/service/types.gen.ts'
import { zHitTestingPayload as zServiceHitTestingPayload } from './generated/api/service/zod.gen.ts'
import { describe, expect, it } from 'vite-plus/test'

type HasExternalRetrievalModel<T> = 'external_retrieval_model' extends keyof T ? true : false

const consoleGenericHasExternalModel: HasExternalRetrievalModel<ConsoleHitTestingPayload> = false
const serviceGenericHasExternalModel: HasExternalRetrievalModel<ServiceHitTestingPayload> = false
const dedicatedExternalHasExternalModel: HasExternalRetrievalModel<ExternalHitTestingPayload> = true

describe('generated hit-testing request contracts', () => {
  it('omits the dead external settings field from generic Console and Service API types', () => {
    expect(consoleGenericHasExternalModel).toBe(false)
    expect(serviceGenericHasExternalModel).toBe(false)
  })

  it('keeps generic payload parsing backward compatible by ignoring the old extra field', () => {
    const legacyPayload = {
      external_retrieval_model: { score_threshold: 0.7, top_k: 3 },
      query: 'query',
    }

    for (const schema of [zConsoleHitTestingPayload, zServiceHitTestingPayload]) {
      expect(schema.parse(legacyPayload)).toEqual({ query: 'query' })
    }
  })

  it('retains external settings on the dedicated external hit-testing contract', () => {
    const payload = {
      external_retrieval_model: {
        score_threshold: 0.7,
        score_threshold_enabled: true,
        top_k: 3,
      },
      query: 'query',
    }

    expect(dedicatedExternalHasExternalModel).toBe(true)
    expect(zExternalHitTestingPayload.parse(payload)).toEqual(payload)
  })
})
