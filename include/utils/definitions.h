#ifndef DEFINITIONS_H
#define DEFINITIONS_H

// Bitmap definitions -> 256 mb
#define BITMAP_BIT_WIDTH 46336
#define BITMAP_INT_WIDTH 1448

// Type definitions for different graphs
using Ordinal = unsigned int;
using Vertex = unsigned int;
using Weight = float;

// 2mb of cache
Ordinal HASH_LIMIT = (1<<19);

// GPU confs
size_t N_STREAMS = 4;
size_t inf = (~(1l<<(sizeof(Ordinal)*8-1)));

#endif // DEFINITIONS_H
