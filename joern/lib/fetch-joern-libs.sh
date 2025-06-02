#!/usr/bin/env bash
set -euo pipefail

# Remove empty jar files
find . -type f -name '*.jar' -size 0 -print -delete

# Target lib dir
LIB_DIR="./"
mkdir -p "$LIB_DIR"

# List of Maven coordinates:
artifacts=(
  "org.openjdk.nashorn:nashorn-core:15.6"
  "org.reflections:reflections:0.9.12"
  "org.json4s:json4s-ast_2.13:3.6.7"
  "org.ow2.asm:asm-util:8.0.1"
  "org.javassist:javassist:3.26.0-GA"
  "org.ow2.asm:asm:8.0.1"
  "org.ow2.asm:asm-tree:8.0.1"
  "org.scala-lang:scala-reflect:2.13.0"
  "org.jline:jline-reader:3.6.2"
  "org.typelevel:cats-effect_2.13:2.0.0"
  "org.jboss.xnio:xnio-nio:3.3.8.Final"
  "org.ow2.asm:asm-commons:8.0.1"
  "org.json4s:json4s-scalap_2.13:3.6.7"
  "org.jline:jline-terminal:3.6.2"
  "org.jetbrains:annotations:13.0"
  "org.jetbrains.kotlin:kotlin-stdlib-common:1.3.72"
  "org.smali:dexlib2:2.4.0"
  "org.slf4j:slf4j-api:1.7.30"
  "org.typelevel:cats-kernel_2.13:2.0.0"
  "org.scala-lang.modules:scala-collection-compat_2.13:2.1.4"
  "org.scala-lang:scala-library:2.13.0"
  "org.scala-lang.modules:scala-java8-compat_2.13:0.9.0"
  "org.scala-lang.modules:scala-xml_2.13:1.2.0"
  "org.jetbrains.kotlin:kotlin-stdlib-jdk8:1.3.72"
  "org.jline:jline-terminal-jna:3.6.2"
  "org.scala-lang.modules:scala-collection-contrib_2.13:0.2.1"
  "org.typelevel:cats-core_2.13:2.0.0"
  "org.jetbrains.kotlin:kotlin-stdlib-jdk7:1.3.72"
  "xmlpull:xmlpull:1.1.3.4d_b4_min"
  "org.jetbrains.kotlin:kotlin-reflect:1.3.72"
  "org.typelevel:jawn-parser_2.13:0.14.2"
  "org.soot-oss:soot:4.2.1"
  "org.msgpack:msgpack-core:0.8.17"
  "org.json4s:json4s-native_2.13:3.6.7"
  "org.lz4:lz4-java:1.7.1"
  "org.json4s:json4s-core_2.13:3.6.7"
  "org.zeroturnaround:zt-zip:1.13"
  "org.scala-lang:scala-compiler:2.13.0"
  "org.jvnet.staxex:stax-ex:1.8"
  "org.typelevel:cats-macros_2.13:2.0.0"
  "org.ow2.asm:asm-analysis:8.0.1"
  "org.scala-lang.modules:scala-parallel-collections_2.13:0.2.0"
  "org.jboss.xnio:xnio-api:3.3.8.Final"
  "org.jetbrains.kotlin:kotlin-stdlib:1.3.72"
)

echo "Fetching ${#artifacts[@]} artifacts into ${LIB_DIR}"
for GAV in "${artifacts[@]}"; do
  echo -n "${GAV}"
  # download into local repo (no transitive deps)
  mvn dependency:get -Dartifact="$GAV" -Dtransitive=false -q
  # copy from m2 to LIB_DIR
  IFS=: read -r groupId artifact version <<<"${GAV}"
  path="${groupId//./\/}/${artifact}/${version}/${artifact}-${version}.jar"
  src="${HOME}/.m2/repository/${path}"
  if [[ -f "${src}" ]]; then
    cp "${src}" "${LIB_DIR}/"
    echo "OK"
  else
    echo "MISSING"
  fi
done
