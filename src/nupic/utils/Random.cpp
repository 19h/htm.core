/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file
    Random Number Generator implementation
*/
#include <iostream> // for istream, ostream

#include <nupic/utils/Log.hpp>
#include <nupic/utils/Random.hpp>
#include <nupic/utils/StringUtils.hpp>

using namespace nupic;

void Random::reseed(UInt64 seed) {
  seed_ = seed;
  gen.seed(seed_);
}


bool Random::operator==(const Random &o) const {
  return seed_ == o.seed_ && \
	 gen == o.gen && \
	 dist_uint_32 == o.dist_uint_32 && \
	 dist_uint_64 == o.dist_uint_64 && \
	 dist_real_64 == o.dist_real_64;
}


Random::Random(UInt64 seed) {
  if (seed == 0) {
    seed_ = 7; //FIXME rd(); //generate random value from HW RNG
  } else {
    seed_ = seed;
  }
  // if seed is zero at this point, there is a logic error.
  NTA_CHECK(seed_ != 0);
  reseed(seed_);
  //distribution ranges
  dist_uint_64 = std::uniform_int_distribution<UInt64>(0, MAX64);
  dist_uint_32 = std::uniform_int_distribution<UInt32>(0, MAX32);
  dist_real_64 = std::uniform_real_distribution<Real64>(0.0, 1.0);
}


namespace nupic {
std::ostream &operator<<(std::ostream &outStream, const Random &r) {
  outStream << "random-v1 ";
  outStream << r.seed_ << " ";
  outStream << r.gen << " ";
  outStream << r.dist_uint_32 << " ";
  outStream << r.dist_uint_64 << " ";
  outStream << r.dist_real_64 << " ";
  outStream << " endrandom-v1 ";
  return outStream;
}


std::istream &operator>>(std::istream &inStream, Random &r) {
  std::string version;

  inStream >> version;
  if (version != "random-v1") {
    NTA_THROW << "Random() deserializer -- found unexpected version string '"
              << version << "'";
  }
  inStream >> r.seed_;

  inStream >> r.dist_uint_32;
  inStream >> r.dist_uint_64;
  inStream >> r.dist_real_64;

  std::string endtag;
  inStream >> endtag;
  if (endtag != "endrandom-v1") {
    NTA_THROW << "Random() deserializer -- found unexpected end tag '" << endtag  << "'";
  }
  inStream.ignore(1);

  return inStream;
}

// helper function for seeding RNGs across the plugin barrier
// Unless there is a logic error, should not be called if
// the Random singleton has not been initialized.
UInt64 GetRandomSeed() {
  Random r = nupic::Random();
  UInt64 result = r.getUInt64();
  return result;
}
} // namespace nupic
