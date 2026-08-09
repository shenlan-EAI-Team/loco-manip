# Future Live Shadow Requirements

The current `G1LiveObservationSource` is a read-only skeleton. It prints:

```text
SHADOW MODE: NO COMMANDS WILL BE SENT
```

Missing interfaces that must be provided explicitly, without guessing topic names or message types:

1. G1 left/right arm feedback reader.
2. G1 waist feedback reader.
3. projected gravity or synchronized IMU reader.
4. left/right O6 feedback readers.
5. D435i `ego_view` RGB reader.
6. cross-source timestamp synchronization.
7. stale data detection and age reporting.

Before any live run:

- `real_hardware_enabled: false`
- `publish_commands: false`
- `shadow_only: true`
- no arm control ownership request
- no G1 publisher and no O6 publisher
- camera/network/SDK latency added to the offline timing budget
- 10 model warmups before timed processing
- synchronization tolerance and stale timeout validated on real streams

The skeleton returns no observation when dependencies are absent and does not fail fatally.
It deliberately contains no command publisher implementation.
