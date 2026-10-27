import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileReader;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Locale;

/**
 * One independent, fixed-physics long-time gate for Formal V1.1.  This class
 * intentionally contains no parameter sweep or model changes beyond V1.1.
 */
public final class New108ReferenceCase24h {
  private static final String ROOT = System.getProperty("az91d.root", ".");
  private static final String OUT = System.getProperty("az91d.out", ROOT + "\\az91d_new108_production");
  private static final String SOURCE = System.getProperty("az91d.source", ROOT + "\\model\\AZ91D_reference_unsolved.mph");
  private static final double ALPHA_MIN = -1.5081273, ALPHA_MAX = -1.3383727;
  private static final double BETA_MIN = -1.6004, BETA_MAX = -1.3416545;
  private New108ReferenceCase24h() {}

  private static double[] eval(Model m, String tag, String kind, String expr) {
    m.result().numerical().create(tag, kind);
    m.result().numerical(tag).set("data", "dset1");
    m.result().numerical(tag).selection().all();
    m.result().numerical(tag).set("expr", expr);
    return m.result().numerical(tag).getReal()[0];
  }

  private static void removeParam(Model m, String name) {
    for (String current : m.param().varnames()) if (current.equals(name)) m.param().remove(name);
  }

  /** Applies exactly the already-audited V1.1 setup to a fresh bootstrap copy. */
  private static void configureV11(Model m, String microFile) throws Exception {
    m.func("int1").importData(microFile);
    m.param().set("T_env", "298.15[K]");
    m.param().set("nacl_wt", "3.5");
    m.param().set("pH0", "7.4");
    removeParam(m, "k_Cl"); removeParam(m, "sigma"); removeParam(m, "sigmae");
    m.param().set("s_alpha", "1.0"); m.param().set("s_beta", "1.0");
    m.param().set("eta_water", "2.414e-5[Pa*s]*10^(247.8/(T_env/1[K]-140))");
    m.param().set("D_temp_scale", "(T_env/298.15[K])*(0.890e-3[Pa*s]/eta_water)");
    m.param().set("pKw_T", "14-0.032*(T_env-298.15[K])/1[K]");
    m.component("comp1").physics("aqt").feature("el1").set("pKw", "pKw_T");
    m.component("comp1").physics("aqt").feature("el1").set("KwType", "userdef");
    m.component("comp1").physics("aqt").feature("el1").set("DH", "9.311e-9[m^2/s]*D_temp_scale");
    m.component("comp1").physics("aqt").feature("el1").set("DOH", "5.273e-9[m^2/s]*D_temp_scale");
    m.component("comp1").physics("aqt").feature("el1").set("minput_temperature_src", "userdef");
    m.component("comp1").physics("aqt").feature("el1").set("minput_temperature", "T_env");
    m.component("comp1").physics("aqt").feature("el1").feature("fds_na").set("D", "DNa_ref*D_temp_scale");
    m.component("comp1").physics("aqt").feature("el1").feature("fds_cl").set("D", "DCl_ref*D_temp_scale");
    m.component("comp1").physics("aqt").feature("el1").feature("fds_mg").set("D", "DMg_ref*D_temp_scale");
    m.component("comp1").physics("aqt").create("hcpce", "HighlyConductivePorousElectrode", 2);
    m.component("comp1").physics("aqt").feature("hcpce").selection().all();
    m.component("comp1").physics("aqt").feature("hcpce").set("item.epsl", "max(pf.Vf1,1e-4)");
    m.component("comp1").physics("aqt").feature("hcpce").set("taul", "1");
    m.component("comp1").physics("aqt").feature("hcpce").set("minput_temperature_src", "userdef");
    m.component("comp1").physics("aqt").feature("hcpce").set("minput_temperature", "T_env");
    m.component("comp1").physics("aqt").feature("hcpce").feature("per1").set("nu0_Mg", -1);
    m.component("comp1").physics("aqt").feature("hcpce").feature("per1").set("ilocmat_mat", "userdef");
    m.component("comp1").physics("aqt").feature("hcpce").feature("per1").set("ilocmat", "s_alpha*i_alpha(-aqt.phil)");
    m.component("comp1").physics("aqt").feature("hcpce").feature("per1").set("Av", "pf.delta*(1-micro(x,y))*(y<=0)");
    m.component("comp1").physics("aqt").feature("hcpce").create("per_beta", "PorousElectrodeReaction", 2);
    m.component("comp1").physics("aqt").feature("hcpce").feature("per_beta").set("nu0OH", 2);
    m.component("comp1").physics("aqt").feature("hcpce").feature("per_beta").set("ilocmat_mat", "userdef");
    m.component("comp1").physics("aqt").feature("hcpce").feature("per_beta").set("ilocmat", "s_beta*i_beta(-aqt.phil)");
    m.component("comp1").physics("aqt").feature("hcpce").feature("per_beta").set("Av", "pf.delta*micro(x,y)*(y<=0)");
    m.component("comp1").variable("var1").set("i_loc", "(s_alpha*i_alpha(-aqt.phil)*(1-micro(x,y))+s_beta*i_beta(-aqt.phil)*micro(x,y))*(y<=0)");
    m.component("comp1").variable("var1").set("Rc", "(1/2)*s_alpha*i_alpha(-aqt.phil)/F_const*(1-micro(x,y))*(y<=0)");
    // V1.1: alpha weighting occurs in Rc once and is not duplicated in Vn.
    m.component("comp1").variable("var1").set("Vn", "Rc*M/rho");
    m.physics("pf").feature("pfm1").set("epsilon_pf", "0.8[um]");
    m.mesh("mesh1").feature("ftri1").feature("size1").set("custom", "on");
    m.mesh("mesh1").feature("ftri1").feature("size1").set("hmax", "2.5[um]");
    m.mesh("mesh1").feature("ftri1").feature("size1").set("hmin", "0.2[um]");
    m.mesh("mesh1").feature("ftri1").feature("size2").set("custom", "on");
    m.mesh("mesh1").feature("ftri1").feature("size2").set("hmax", "0.25[um]");
    m.mesh("mesh1").feature("ftri1").feature("size2").set("hmin", "0.05[um]");
    m.mesh("mesh1").run(); m.sol().remove("sol1");
    m.study("std1").feature("time").set("tunit", "s");
    m.study("std1").feature("time").set("tlist", "0 900 1800 3600 7200 10800 14400 18000 21600 25200 28800 32400 36000 39600 43200 46800 50400 54000 57600 61200 64800 68400 72000 75600 79200 82800 86400");
  }

  private static double crossing(double[] field, int ix, int nx, int ny, double ymin, double ymax, double reference) {
    double bestY = Double.NaN, bestDistance = Double.POSITIVE_INFINITY;
    for (int iy = 0; iy < ny - 1; iy++) {
      double a = field[iy * nx + ix], b = field[(iy + 1) * nx + ix];
      if ((a - 0.5) * (b - 0.5) > 0 || a == b) continue;
      double y0 = ymin + (ymax - ymin) * iy / (ny - 1.0);
      double y1 = ymin + (ymax - ymin) * (iy + 1) / (ny - 1.0);
      double yc = y0 + (0.5 - a) * (y1 - y0) / (b - a);
      double distance = Double.isNaN(reference) ? Math.abs(yc) : Math.abs(yc - reference);
      if (distance < bestDistance) { bestDistance = distance; bestY = yc; }
    }
    return bestY;
  }

  private static double percentile(double[] values, int n, double q) {
    if (n == 0) return Double.NaN;
    double[] c = Arrays.copyOf(values, n); Arrays.sort(c);
    double p = q * (n - 1);
    int lo = Math.max(0, (int)Math.floor(p));
    int hi = Math.min(n - 1, lo + 1);
    return c[lo] + (p - lo) * (c[hi] - c[lo]);
  }

  /** High-resolution FEM interpolation is retained only long enough to calculate scalar label evidence. */
  /* rows: mean depth, p95 depth, max depth, relative L2, max abs, mean abs */
  private static double[][] phaseStats(Model m, double[] times) throws Exception {
    final int nx = 1024, ny = 1024, n = nx * ny;
    double[][] v = m.mesh("mesh1").getVertex();
    double xmin = Double.POSITIVE_INFINITY, xmax = Double.NEGATIVE_INFINITY, ymin = Double.POSITIVE_INFINITY, ymax = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < v[0].length; i++) { xmin=Math.min(xmin,v[0][i]); xmax=Math.max(xmax,v[0][i]); ymin=Math.min(ymin,v[1][i]); ymax=Math.max(ymax,v[1][i]); }
    double[][] coord = new double[2][n];
    for (int iy=0; iy<ny; iy++) for (int ix=0; ix<nx; ix++) { int p=iy*nx+ix; coord[0][p]=xmin+(xmax-xmin)*ix/(nx-1.0); coord[1][p]=ymin+(ymax-ymin)*iy/(ny-1.0); }
    m.result().numerical().create("longphase", "Interp");
    m.result().numerical("longphase").set("data", "dset1");
    m.result().numerical("longphase").set("expr", new String[] {"pf.Vf1"});
    m.result().numerical("longphase").setInterpolationCoordinates(coord);
    double[][][] raw = m.result().numerical("longphase").getData();
    if (raw.length != 1 || raw[0].length != times.length) throw new Exception("Unexpected phase interpolation shape");
    int nt=times.length;
    double[][] s=new double[6][nt];
    double[] initial=raw[0][0], initialInterface=new double[nx];
    for (int ix=0; ix<nx; ix++) initialInterface[ix]=crossing(initial,ix,nx,ny,ymin,ymax,Double.NaN);
    double initialNorm=0; for(double x:initial) initialNorm+=x*x;
    for (int it=0; it<nt; it++) {
      double[] f=raw[0][it], depth=new double[nx]; int valid=0; double sumD=0, maxD=Double.NEGATIVE_INFINITY, l2=0, absSum=0, maxAbs=0;
      for (int p=0;p<n;p++) { double d=f[p]-initial[p]; l2+=d*d; absSum+=Math.abs(d); maxAbs=Math.max(maxAbs,Math.abs(d)); }
      for (int ix=0;ix<nx;ix++) if (!Double.isNaN(initialInterface[ix])) { double now=crossing(f,ix,nx,ny,ymin,ymax,initialInterface[ix]); if(!Double.isNaN(now)) { double d=initialInterface[ix]-now; depth[valid++]=d; sumD+=d; maxD=Math.max(maxD,d); } }
      s[0][it]=valid>0?sumD/valid:Double.NaN; s[1][it]=percentile(depth,valid,0.95); s[2][it]=valid>0?maxD:Double.NaN;
      s[3][it]=Math.sqrt(l2/Math.max(initialNorm,1e-300)); s[4][it]=maxAbs; s[5][it]=absSum/n;
    }
    m.result().numerical().remove("longphase");
    return s;
  }

  private static String rangeStatus(double fraction) { return fraction < 1e-6 ? "inside" : (fraction < 0.5 ? "partially outside" : "substantially outside"); }

  /** Writes fields on a fixed physical-coordinate grid; the aspect ratio is taken from the mesh bounds. */
  private static void exportFields(Model m, String file) throws Exception {
    final int nx = 512;
    double[][] v = m.mesh("mesh1").getVertex();
    double xmin=Double.POSITIVE_INFINITY,xmax=Double.NEGATIVE_INFINITY,ymin=Double.POSITIVE_INFINITY,ymax=Double.NEGATIVE_INFINITY;
    for (int i=0;i<v[0].length;i++) { xmin=Math.min(xmin,v[0][i]); xmax=Math.max(xmax,v[0][i]); ymin=Math.min(ymin,v[1][i]); ymax=Math.max(ymax,v[1][i]); }
    final int ny=Math.max(2,(int)Math.round(nx*(ymax-ymin)/(xmax-xmin)));
    double[][] coord=new double[2][nx*ny];
    for(int iy=0;iy<ny;iy++) for(int ix=0;ix<nx;ix++){int p=iy*nx+ix;coord[0][p]=xmin+(xmax-xmin)*ix/(nx-1.0);coord[1][p]=ymin+(ymax-ymin)*iy/(ny-1.0);}
    String tag="field_export";
    m.result().numerical().create(tag,"Interp"); m.result().numerical(tag).set("data","dset1");
    m.result().numerical(tag).set("expr",new String[]{"aqt.phil","pf.delta*(1-micro(x,y))*s_alpha*i_alpha(-aqt.phil)","pf.delta*micro(x,y)*s_beta*i_beta(-aqt.phil)","pf.Vf1","aqt.c_Mg","aqt.pH"});
    m.result().numerical(tag).setInterpolationCoordinates(coord);
    double[][][] data=m.result().numerical(tag).getData(); double[] times=m.sol("sol1").getPVals();
    try(DataOutputStream out=new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))){
      out.writeInt(0x415A5632); out.writeInt(3); out.writeInt(nx); out.writeInt(ny); out.writeInt(data.length); out.writeInt(times.length);
      out.writeDouble(xmin);out.writeDouble(xmax);out.writeDouble(ymin);out.writeDouble(ymax);
      for(double time:times)out.writeDouble(time);
      for(int e=0;e<data.length;e++)for(int it=0;it<data[e].length;it++)for(double value:data[e][it])out.writeDouble(value);
    }
    m.result().numerical().remove(tag);
  }

  public static void runCase(String label, String microFile) throws Exception {
    final String caseDir=OUT+"\\comsol_runs\\"+label;
    new java.io.File(caseDir).mkdirs();
    final String csv=caseDir+"\\metrics.csv";
    final String fieldFile=caseDir+"\\fields.bin";
    long allStart=System.nanoTime(); Model m=ModelUtil.load("FormalV11Dataset24h_"+label, SOURCE);
    try {
      configureV11(m,microFile);
      long solveStart=System.nanoTime();
      m.study("std1").run();
      double wall=(System.nanoTime()-solveStart)/1e9;
      double[] t=m.sol("sol1").getPVals();
      if (t.length != 27 || Math.abs(t[t.length-1]-86400)>1e-8) throw new Exception("24 h horizon gate did not return all requested outputs");
      double[] ia=eval(m,"ia","IntSurface","pf.delta*(1-micro(x,y))*(y<=0)*s_alpha*i_alpha(-aqt.phil)");
      double[] ib=eval(m,"ib","IntSurface","pf.delta*micro(x,y)*(y<=0)*s_beta*i_beta(-aqt.phil)");
      double[] mg=eval(m,"mg","IntSurface","max(pf.Vf1,1e-4)*aqt.c_Mg");
      double[] metal=eval(m,"metal","IntSurface","pf.Vf2*(y<=0)");
      double[] minpH=eval(m,"minpH","MinSurface","aqt.pH"), maxpH=eval(m,"maxpH","MaxSurface","aqt.pH");
      double[] maxMg=eval(m,"maxMg","MaxSurface","aqt.c_Mg"), minNa=eval(m,"minNa","MinSurface","aqt.c_Na"), minCl=eval(m,"minCl","MinSurface","aqt.c_Cl"), minMg=eval(m,"minMg","MinSurface","aqt.c_Mg");
      double[] alphaMin=eval(m,"alphaMin","MinSurface","if((1-micro(x,y))*(y<=0)>0.5,-aqt.phil,1e30[V])"), alphaMax=eval(m,"alphaMax","MaxSurface","if((1-micro(x,y))*(y<=0)>0.5,-aqt.phil,-1e30[V])");
      double[] betaMin=eval(m,"betaMin","MinSurface","if(micro(x,y)*(y<=0)>0.5,-aqt.phil,1e30[V])"), betaMax=eval(m,"betaMax","MaxSurface","if(micro(x,y)*(y<=0)>0.5,-aqt.phil,-1e30[V])");
      double[] aa=eval(m,"alphaArea","IntSurface","pf.delta*(1-micro(x,y))*(y<=0)"), ba=eval(m,"betaArea","IntSurface","pf.delta*micro(x,y)*(y<=0)");
      double[] ao=eval(m,"alphaOutside","IntSurface","pf.delta*(1-micro(x,y))*(y<=0)*if((-aqt.phil)<-1.5081273[V]||(-aqt.phil)>-1.3383727[V],1,0)"), bo=eval(m,"betaOutside","IntSurface","pf.delta*micro(x,y)*(y<=0)*if((-aqt.phil)<-1.6004[V]||(-aqt.phil)>-1.3416545[V],1,0)");
      double[][] ps=phaseStats(m,t);
      double rho=m.param().evaluate("rho"), M=m.param().evaluate("M"), F=m.param().evaluate("F_const");
      try (PrintWriter out=new PrintWriter(csv,"UTF-8")) {
        out.println("time_s,I_alpha_A_per_m,I_beta_A_per_m,current_closure_error,Faraday_alpha_mol_per_m,Faraday_mass_kg_per_m,continuous_phase_mass_kg_per_m,phase_Faraday_relative_difference,Mg_inventory_mol_per_m,Mg_Faraday_conservation_error,pH_min,pH_max,Mg2_max_mol_per_m3,Na_min_mol_per_m3,Cl_min_mol_per_m3,concentration_nonnegative,alpha_potential_min_V,alpha_potential_max_V,beta_potential_min_V,beta_potential_max_V,alpha_outside_fraction,beta_outside_fraction,alpha_range_status,beta_range_status,mean_depth_m,p95_depth_m,max_depth_m,phase_relative_L2_change,phase_max_absolute_change,phase_mean_absolute_change,solve_wall_total_s");
        double q=0, metal0=metal[0], mg0=mg[0];
        for(int i=0;i<t.length;i++) { if(i>0) q+=0.5*(ia[i-1]+ia[i])*(t[i]-t[i-1]); double fm=q/(2*F), fmass=fm*M, pmass=(metal0-metal[i])*rho, closure=(Math.abs(ia[i])+Math.abs(ib[i]))>0?Math.abs(ia[i]+ib[i])/(Math.abs(ia[i])+Math.abs(ib[i])):0, mgerr=Math.abs((mg[i]-mg0)-fm)/Math.max(Math.abs(fm),1e-30), perr=Math.abs(pmass-fmass)/Math.max(Math.abs(fmass),1e-30), aof=ao[i]/Math.max(aa[i],1e-30), bof=bo[i]/Math.max(ba[i],1e-30); boolean nonnegative=minNa[i]>=0&&minCl[i]>=0&&minMg[i]>=0;
          out.printf(Locale.US,"%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g%n",t[i],ia[i],ib[i],closure,fm,fmass,pmass,perr,mg[i],mgerr,minpH[i],maxpH[i],maxMg[i],minNa[i],minCl[i],Boolean.toString(nonnegative),alphaMin[i],alphaMax[i],betaMin[i],betaMax[i],aof,bof,rangeStatus(aof),rangeStatus(bof),ps[0][i],ps[1][i],ps[2][i],ps[3][i],ps[4][i],ps[5][i],wall);
        }
      }
      exportFields(m,fieldFile);
      System.out.printf(Locale.US,"FORMAL_V11_108=completed;case=%s;solve_wall_s=%.6g;total_wall_s=%.6g;csv=%s;fields=%s;final_time_s=%.12g%n",label,wall,(System.nanoTime()-allStart)/1e9,csv,fieldFile,t[t.length-1]);
    } catch (Exception failure) {
      // Best-effort preservation of the current COMSOL state. It may contain
      // the last accepted transient solution when the solver aborts.
      try { m.save(caseDir+"\\last_accepted_or_failure_state.mph"); }
      catch (Exception ignored) { System.err.println("Could not save failure state: "+ignored.getMessage()); }
      throw failure;
    } finally { ModelUtil.remove("FormalV11Dataset24h_"+label); }
  }

  public static void main(String[] args) throws Exception {
    String caseFile = System.getProperty("az91d.caseFile", OUT + "\\scheduler\\CASE_SLOT_1.txt");
    String label, micro;
    try (BufferedReader reader = new BufferedReader(new FileReader(caseFile))) {
      label = reader.readLine();
      micro = reader.readLine();
    }
    if (label == null || micro == null) throw new IllegalArgumentException("Invalid production case file");
    runCase(label.trim(), micro.trim());
  }
}
