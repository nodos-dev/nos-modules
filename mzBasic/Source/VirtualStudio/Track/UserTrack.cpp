// Copyright MediaZ AS. All Rights Reserved.

#include "BasicMain.h"
#include "Args.h"
#include "Builtins_generated.h"
#include "glm/gtx/euler_angles.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/glm.hpp"

namespace mz
{

void RegisterController(NodeActionsMap& functions)
{
    auto& actions = functions["mz.Controller"];
    
    struct Controller {
         //union
         //{
         //   struct
         //   {
         //       glm::dvec3 r;
         //       glm::dvec3 u;
         //       glm::dvec3 f;
         //   };
         //   glm::dmat3 ori;
         //};

        // Controller() : r(0, 1, 0), u(0, 0, 1), f(1, 0, 0) {}
        
        glm::dvec3 p   = glm::vec3(0, 0, 5);
        f64 yaw = 0;
        f64 pitch = 0;
        glm::dvec3 v{};

        f64 impulse = 1.0;
        f64 decay = 0;

        mz::fb::UUID id;
        f64 fov = 90.0;
        bool zoom = false;

        void fill(fb::Track& track)
        {
            // Assumes fields are present in the table.
            auto* rot = track.mutable_rotation();
            assert(rot);
            rot->mutate_y(glm::degrees(pitch));
            rot->mutate_z(glm::degrees(yaw));
            rot->mutate_x(0);
            auto* loc = track.mutable_location();
            MZ_CHECK(loc)
            loc->mutate_x(p.x);
            loc->mutate_y(p.y);
            loc->mutate_z(p.z);
            MZ_CHECK(track.mutate_fov(zoom ? 30.f : 74.f))
            MZ_CHECK(track.mutate_distortion_scale(1))
            MZ_CHECK(track.mutate_render_ratio(1))
            MZ_CHECK(track.mutate_pixel_aspect_ratio(1))
            auto* sensorSize = track.mutable_sensor_size();
            MZ_CHECK(sensorSize)
            *sensorSize = fb::vec2d(9.590, 5.394);
        }
    };

    actions.NodeCreated = [](fb::Node const& n, mz::Args& args, void** ctx) {
        auto c = new Controller();
        *ctx = c;
        c->id = *n.pins()->begin()->id();
        auto trackBuffer = args.GetBuffer("Track");
        auto track = trackBuffer->As<mz::fb::Track>();
        c->p = (glm::dvec3&)(*track->mutable_location());
        auto rot = (glm::dvec3&)(*track->mutable_rotation());
        c->pitch  = glm::radians(rot.y);
        c->yaw    = glm::radians(rot.z);
        c->fov    = track->fov();
        c->fill(*track);
        //flatbuffers::FlatBufferBuilder fbb;
        //std::vector<flatbuffers::Offset<mz::fb::NodeStatusMessage>> msg;
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Hold RMB to start controlling", fb::NodeStatusMessageType::INFO));
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Use your mouse to look around", fb::NodeStatusMessageType::INFO));
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "WASD to move", fb::NodeStatusMessageType::INFO));
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Space to reset view", fb::NodeStatusMessageType::INFO));
        //msg.push_back(fb::CreateNodeStatusMessageDirect(fbb, "Good luck have fun", fb::NodeStatusMessageType::INFO));
        //services.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, n.id(), ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &msg)));
    };

    actions.NodeRemoved = [](void* ctx, auto&){delete (Controller*)ctx;};
    actions.KeyEvent = [](void* ctx, u64 key, u64 button, mz::fb::vec2 const& mdelta, u32 action){
        auto c = (Controller*)ctx;
        // c->fov += 2 *(((key == 'F') - (key == 'G')));
        // c->m *= 1.0 + 0.2 * ((key == 'O') - (key == 'P'));
        if (32 == key)
        {
            c->pitch = 0;
            c->yaw   = 180;
            c->p = glm::vec3{600, 50, 100};
            flatbuffers::FlatBufferBuilder fbb;
            GServices.HandleEvent(CreateAppEvent(fbb, mz::app::CreatePinDirtied(fbb, &c->id)));
        }

        glm::dvec2 ANG = glm::dvec2(c->yaw, c->pitch);
        glm::dvec2 COS = cos(ANG);
        glm::dvec2 SIN = sin(ANG);

        glm::dvec3 f = glm::dvec3(COS.y * COS.x, COS.y * SIN.x, SIN.y);
        glm::dvec3 r = glm::dvec3(-SIN.x, COS.x, 0);
        glm::dvec3 u = cross(f, r);
        
        glm::dvec3 pf =  f * f64((key == 'W') - (key == 'S'));
        glm::dvec3 pr =  r * f64((key == 'D') - (key == 'A'));
        glm::dvec3 pu =  u * f64((key == 'E') - (key == 'Q'));
        
        c->v += c->impulse * (pf + pr + pu);
        c->p += 10.0 * (pf + pr + pu);

        // c->zoom ^= (action != 2 && button == 1);
        c->pitch -= mdelta.y() * 0.015;
        c->yaw   += mdelta.x() * 0.015;
        
        c->yaw   = fmod(c->yaw, glm::radians(360.));
        // c->pitch = glm::clamp(c->pitch, glm::radians(-80.), glm::radians(80.));
        
        flatbuffers::FlatBufferBuilder fbb;
        GServices.HandleEvent(CreateAppEvent(fbb, mz::app::CreatePinDirtied(fbb, &c->id)));
    };

    actions.EntryPoint = [](mz::Args& args, void* ctx){
        auto c = (Controller*)ctx;

        c->impulse = glm::max(*args.Get<f64>("Impulse"), 1.);
        c->decay   = glm::max(*args.Get<f64>("Decay"), 0.);
        c->p += c->v;
        c->v *= exp(-c->decay);
        auto trackBuffer = args.GetBuffer("Track");
        auto track = trackBuffer->As<mz::fb::Track>();
        c->fill(*track);
        return true;
    };
}

}