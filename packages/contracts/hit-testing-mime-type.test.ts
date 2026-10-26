import type { HitTestingFile as ConsoleHitTestingFile } from './generated/api/console/datasets/types.gen.ts'
import { datasets as consoleDatasets } from './generated/api/console/datasets/orpc.gen.ts'
import { zHitTestingFile as zConsoleHitTestingFile } from './generated/api/console/datasets/zod.gen.ts'
import type { HitTestingFile as ServiceHitTestingFile } from './generated/api/service/types.gen.ts'
import { datasets as serviceDatasets } from './generated/api/service/orpc.gen.ts'
import { zHitTestingFile as zServiceHitTestingFile } from './generated/api/service/zod.gen.ts'
import { describe, expect, it } from 'vite-plus/test'

const fileWithoutRecordedMimeType = {
  extension: 'bin',
  id: 'file-1',
  mime_type: null,
  name: 'attachment.bin',
  size: 4,
  source_url: 'https://example.com/attachment.bin',
} satisfies ConsoleHitTestingFile & ServiceHitTestingFile

describe('generated hit-testing file contract', () => {
  it('keeps MIME type required but nullable on Console and Service API responses', () => {
    for (const schema of [zConsoleHitTestingFile, zServiceHitTestingFile]) {
      expect(schema.parse(fileWithoutRecordedMimeType)).toEqual(fileWithoutRecordedMimeType)
      expect(schema.safeParse({ ...fileWithoutRecordedMimeType, mime_type: 'application/octet-stream' }).success).toBe(
        true,
      )
      const { mime_type: _mimeType, ...missingMimeType } = fileWithoutRecordedMimeType
      expect(schema.safeParse(missingMimeType).success).toBe(false)
    }
  })

  it('keeps the nullable response reachable through both generated oRPC routers', () => {
    expect(consoleDatasets.byDatasetId.hitTesting.post).toBeDefined()
    expect(serviceDatasets.byDatasetId.hitTesting.post).toBeDefined()
    expect(serviceDatasets.byDatasetId.retrieve.post).toBeDefined()
  })
})
