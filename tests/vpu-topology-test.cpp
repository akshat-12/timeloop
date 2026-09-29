// Run with a processed native architecture YAML containing a VPU and a problem.
#include "model/topology.hpp"
#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
  assert(argc == 2);
  config::CompoundConfig config{argv[1]};
  auto root = config.getRoot();
  problem::Workload workload;
  problem::ParseWorkload(root.lookup("problem"), workload);
  auto specs = model::Topology::ParseTreeSpecs(root.lookup("architecture"), false);
  assert(specs.vpu && specs.vpu->name.Get() == "VPU");
  assert(specs.GetArithmeticLevel()->name.Get() == "mac");
  auto copy = specs;
  assert(copy.vpu != specs.vpu); // No shared mutable specs between candidates.
  model::Topology::Specs assigned;
  assigned = specs;
  assert(assigned.vpu && assigned.vpu != specs.vpu);
  model::Topology topology;
  topology.Spec(assigned);
  assert(topology.ViewVPU() && topology.ViewVPU()->Cycles() == 0);
  assert(topology.ViewArithmeticLevel()->Name() == "mac");
  auto clone = topology;
  assert(clone.ViewVPU() && clone.ViewVPU() != topology.ViewVPU());
  model::Topology another;
  another = topology;
  assert(another.ViewVPU() && another.ViewVPU() != topology.ViewVPU());
  assigned.vpu.reset();
  topology.Spec(assigned);
  assert(!topology.ViewVPU()); // Re-specification must not leave a stale VPU.
  std::cout << "VPU topology parsing, construction, copy, assignment, and removal passed\n";
}
