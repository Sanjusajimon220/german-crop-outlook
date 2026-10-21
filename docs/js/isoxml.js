// ISO 11783-10 task data with one product and a type-2 grid (rate per cell). Pure; tested with node.
// zones.classes are in ISO grid order: row 0 = southernmost row, columns west -> east; -1 = outside the field.
// rates: kg/ha per zone class [class 0, class 1, class 2]; written as mg/m2 (kg/ha x 100), DDI 0006.

export function buildTaskData({ field, zones, rates, grid, product = "Fertiliser (rate set by user)" }) {
  const { rows, cols } = grid;
  const bin = new Uint8Array(rows * cols * 4);
  const view = new DataView(bin.buffer);
  for (let i = 0; i < rows * cols; i++) {
    const c = zones.classes[i];
    view.setInt32(4 * i, c >= 0 ? Math.round(rates[c] * 100) : 0, true);
  }
  const def = Math.round(Math.max(...rates) * 100);
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<ISO11783_TaskData VersionMajor="4" VersionMinor="2" ManagementSoftwareManufacturer="Groundtruth-earth" ManagementSoftwareVersion="1.0" DataTransferOrigin="1">
<PDT A="PDT1" B="${product.replace(/[<>&"]/g, "")}"/>
<TSK A="TSK1" B="Field ${field.id} zones" G="1">
<TZN A="1" B="grid">
<PDV A="0006" B="${def}" C="PDT1"/>
</TZN>
<GRD A="${grid.minLat.toFixed(9)}" B="${grid.minLon.toFixed(9)}" C="${grid.dLat.toFixed(9)}" D="${grid.dLon.toFixed(9)}" E="${cols}" F="${rows}" G="GRD00001" H="${bin.length}" I="2" J="1"/>
</TSK>
</ISO11783_TaskData>
`;
  return { xml, bin };
}
