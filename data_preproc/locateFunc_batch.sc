@main def exec(worklistFile: String) = {
  scala.io.Source.fromFile(worklistFile).getLines().foreach { line =>
    val parts = line.split(",", 2)
    if (parts.length == 2) {
      val inputFile = parts(0).trim
      val outFile = parts(1).trim
      try {
        importCode(inputFile)
        cpg.method.map(x=>(x.fullName,x.filename,x.lineNumber,x.lineNumberEnd)).toJson |> outFile
        delete
      } catch {
        case e: Exception =>
          System.err.println(s"Error processing $inputFile: ${e.getMessage}")
      }
    }
  }
}
