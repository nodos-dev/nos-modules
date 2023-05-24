
#define SHARP_EDGES_BIT 0
#define HAS_LEFT_WING_BIT 1
#define HAS_RIGHT_WING_BIT 2

bool GetBit(uint flags, uint bit)
{
    return 0 != ((flags >> bit) & 1);
}
