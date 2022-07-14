#include "mzTextOverlayConfig.h"

#include "Args.h"

#include "mzTextOverlay.pb.h"

extern "C"
{

typedef struct {float x, y;} vec2;
typedef struct {float x, y, z, w;} vec4;

extern const struct
{
    unsigned short x, y, w, h, ox, oy;
} font_layout[];


static void process_char(mz::text::VertexInput *glyph, char c, vec2 tex, vec2 scale, vec2 screen_pos)
{
    const float p0x = font_layout[c].x / tex.x;
    const float p0y = font_layout[c].y / tex.y;

    const float p1x = font_layout[c].w / tex.x + p0x;
    const float p1y = font_layout[c].h / tex.y + p0y;

    const float c0x = font_layout[c].ox * scale.x + screen_pos.x;
    const float c0y = font_layout[c].oy * scale.y + screen_pos.y;
    
    const float c1x = font_layout[c].w * scale.x + c0x;
    const float c1y = font_layout[c].h * scale.y + c0y;

    vec4 dat[6] = {{p0x, p1y, c0x, c0y},
                   {p1x, p1y, c1x, c0y},
                   {p0x, p0y, c0x, c1y},
                   {p0x, p0y, c0x, c1y},
                   {p1x, p1y, c1x, c0y},
                   {p1x, p0y, c1x, c1y}};

    for(auto v: dat)
    {
        auto in = glyph->add_input();
        in->mutable_pos()->set_x(v.x);
        in->mutable_pos()->set_y(v.y);
        in->mutable_uv()->set_x(v.z);
        in->mutable_uv()->set_y(v.w);
    }
}

void mzTextOverlay_API ProcessString(void** inout, const char* metaData)
{
	mz::Args params(inout, metaData);
    // const char* text = params.Get<const char*>(0);
    // mz::text::VertexInput* input;
    // const char* text = params.Get<const char*>(1);

    // for(auto c : std::string(text))
    // {
    //     process_char(glyph, fmtbuf[i], atlas.size, scale, pos, color);
    // }
}

}
