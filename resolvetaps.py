# Volatility
# Copyright (C) 2007,2008 Volatile Systems
# Copyright (c) 2011 Brendan Dolan-Gavitt <brendandg@gatech.edu>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details. 
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
#

import os
import errno
import bisect
import shutil
import urllib
import urllib2
import subprocess
import tempfile
import volatility.win32.tasks as tasks
import volatility.win32.modules as modules
import volatility.plugins.dlldump as dlldump
import volatility.plugins.moddump as moddump
import volatility.obj as obj
import volatility.debug as debug
import volatility.utils as utils

# From PDBParse
import pdbparse.undname as und
import pdbparse.peinfo as symchk
import pdbparse.symlookup as lookup

IMAGE_DIRECTORY_ENTRY_EXPORT          = 0   # Export Directory
IMAGE_DIRECTORY_ENTRY_IMPORT          = 1   # Import Directory
IMAGE_DIRECTORY_ENTRY_RESOURCE        = 2   # Resource Directory
IMAGE_DIRECTORY_ENTRY_EXCEPTION       = 3   # Exception Directory
IMAGE_DIRECTORY_ENTRY_SECURITY        = 4   # Security Directory
IMAGE_DIRECTORY_ENTRY_BASERELOC       = 5   # Base Relocation Table
IMAGE_DIRECTORY_ENTRY_DEBUG           = 6   # Debug Directory
IMAGE_DIRECTORY_ENTRY_ARCHITECTURE    = 7   # Architecture Specific Data
IMAGE_DIRECTORY_ENTRY_GLOBALPTR       = 8   # RVA of GP
IMAGE_DIRECTORY_ENTRY_TLS             = 9   # TLS Directory
IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG     = 10   # Load Configuration Directory
IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT    = 11   # Bound Import Directory in headers
IMAGE_DIRECTORY_ENTRY_IAT             = 12   # Import Address Table
IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT    = 13   # Delay Load Import Descriptors
IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR  = 14   # COM Runtime descriptor

debug_types = dict(
    IMAGE_DEBUG_TYPE_UNKNOWN              = 0,
    IMAGE_DEBUG_TYPE_COFF                 = 1,
    IMAGE_DEBUG_TYPE_CODEVIEW             = 2,
    IMAGE_DEBUG_TYPE_FPO                  = 3,
    IMAGE_DEBUG_TYPE_MISC                 = 4,
    IMAGE_DEBUG_TYPE_EXCEPTION            = 5,
    IMAGE_DEBUG_TYPE_FIXUP                = 6,
    IMAGE_DEBUG_TYPE_OMAP_TO_SRC          = 7,
    IMAGE_DEBUG_TYPE_OMAP_FROM_SRC        = 8,
    IMAGE_DEBUG_TYPE_BORLAND              = 9,
    IMAGE_DEBUG_TYPE_RESERVED10           = 10,
    IMAGE_DEBUG_TYPE_CLSID                = 11,
)
debug_types.update((v,k) for k,v in debug_types.items())

sympath = [
    "http://msdl.microsoft.com/download/symbols/%s/%s/",
    "http://symbols.mozilla.org/firefox/%s/%s/",
    'http://chromium-browser-symsrv.commondatastorage.googleapis.com/%s/%s/',
]

# From symchk.py in pdbparse
class SymDownloader(urllib.FancyURLopener):
    version = "Microsoft-Symbol-Server/6.6.0007.5"
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        if errcode == 404:
            raise urllib2.HTTPError(url, errcode, errmsg, headers, fp)
        else:
            FancyURLopener.http_error_default(url, fp, errcode, errmsg, headers)

# From symchk.py in pdbparse
def download_file(guid,fname,path=""):
    ''' 
    Download the symbols specified by guid and filename. Note that 'guid'
    must be the GUID from the executable with the dashes removed *AND* the
    Age field appended. The resulting file will be saved to the path argument,
    which defaults to the current directory.
    '''

    for url in sympath:
        url = url % (fname,guid)
        
        # Most files are cab-compressed, so we try the _ variant first
        tries = [ fname[:-1] + '_', fname ]

        for t in tries:
            outfile = os.path.join(path,t)
            try:
                SymDownloader().retrieve(url+t, outfile)
                return outfile
            except urllib2.HTTPError, e:
                if e.code != 404:
                    raise
                else:
                    pass
    return None

class ResolveTaps(dlldump.DLLDump):
    """Check the integrity of an executable or DLL in memory"""

    def __init__(self, config, *args):
        dlldump.DLLDump.__init__(self, config, *args)
        config.remove_option("DUMP-DIR")
        config.remove_option("UNSAFE")
        config.remove_option("BASE")
        config.add_option('BASE', short_option = 'b',
                      help = 'Check DLL at BASE address in the process address space',
                      action = 'store', type = 'int')
        config.add_option('TAP_FILE', short_option = 't',
                      help = 'Obtain tap points from TAP_FILE',
                      action = 'store', type = 'str')

    def get_file_by_guid(self, exe_base, guid):
        outdir = os.path.abspath(
            os.path.join(self._config.CACHE_DIRECTORY, "vol_symcache", exe_base, guid)
        )
        
        # Try to get it from the cache to avoid hammering MS's symbol server
        if os.path.exists(os.path.join(outdir, exe_base)):
            exename = os.path.join(outdir, exe_base)
            debug.debug("Copy of {0} found in cache: {1}".format(exe_base, exename))
            return exename

        # mkdir -p outdir
        try:
            os.makedirs(outdir)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
            else: raise

        outfile = download_file(guid, exe_base, path=outdir)
    
        if not outfile:
            raise ValueError("Failed to download executable")
        else:
            debug.debug("Copy of {0} saved to output file {1}".format(exe_base, outfile))

        exename = outfile
        if outfile.endswith("_"):
            p = subprocess.Popen(['cabextract', '-d'+outdir, outfile], stdout=subprocess.PIPE)
            cabout = p.communicate()[0]
            for line in cabout.splitlines():
                line = line.strip()
                if line.startswith('extracting'):
                    exename = line.split()[-1]
                    break
            else:
                raise ValueError("Couldn't extract executable from cabinet")

            # Get rid of the cabinet file
            os.remove(outfile)

        debug.debug("Done extracting cabinet file")

        # Rename so we can find it in the cache next time
        dirname = os.path.dirname(exename)
        basename = os.path.basename(exename)
        newname = os.path.join(dirname, basename.lower())
        debug.debug("Renaming {0} to {1}".format(exename, newname))
        os.rename(exename, newname)
        exename = newname
        
        return exename

    def get_pdb_mem(self, ps_ad, mod_name, mod_base, from_file=False):
        """Attempts to retrieve the debug section from a PE in memory"""
        dos_hdr = obj.Object('_IMAGE_DOS_HEADER', mod_base, ps_ad)
        nt_hdrs = obj.Object('_IMAGE_NT_HEADERS', mod_base + dos_hdr.e_lfanew, ps_ad)
        debug_dir_ent = nt_hdrs.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG]
        if not debug_dir_ent.VirtualAddress:
            return False,""

        debug_dir_addr = mod_base+debug_dir_ent.VirtualAddress
        debug_dir = obj.Object('_IMAGE_DEBUG_DIRECTORY', debug_dir_addr, ps_ad)
        if not debug_dir.AddressOfRawData:
            return False,""
        
        debug_data_addr = mod_base+debug_dir.AddressOfRawData

        debug_data = ps_ad.read(debug_data_addr, debug_dir.SizeOfData)
        if not debug_data:
            return False,""
        return debug_data, debug_types.get(debug_dir.Type.v(), "IMAGE_DEBUG_TYPE_UNKNOWN")

    def get_pe(self, ps_ad, mod_name, mod_base):
        """Retrieves a clean copy of a module from MS's symbol server.
        It should go without saying that this only works for MS executables."""
        
        dos_hdr = obj.Object('_IMAGE_DOS_HEADER', mod_base, ps_ad)
        nt_hdrs = obj.Object('_IMAGE_NT_HEADERS', mod_base + dos_hdr.e_lfanew, ps_ad)
        guid = "{0:x}{1:x}".format(nt_hdrs.FileHeader.TimeDateStamp,
                                   nt_hdrs.OptionalHeader.SizeOfImage)
        exe_base = str(mod_name).lower()

        exename = self.get_file_by_guid(exe_base, guid)

        # We should have a clean copy of the executable by now
        return exename

    def get_pdb(self, ps_ad, mod_name, mod_base):
        try:
            dbgdata, tp = self.get_pdb_mem(ps_ad, mod_name, mod_base)
            # Try to get the PE and then get the PDB from that
            if not dbgdata:
                exename = self.get_pe(ps_ad, mod_name, mod_base)
                dbgdata, tp = symchk.get_pe_debug_data(exename)

            if tp == "IMAGE_DEBUG_TYPE_CODEVIEW":
                # XP+
                if dbgdata[:4] == "RSDS":
                    (guid,filename) = symchk.get_rsds(dbgdata)
                elif dbgdata[:4] == "NB10":
                    (guid,filename) = symchk.get_nb10(dbgdata)
                else:
                    debug.warning("ERR: CodeView section not NB10 or RSDS")
                    return
                guid = guid.upper()
                pdbname = self.get_file_by_guid(filename, guid)
                return pdbname
        except ValueError:
            return None

    def format_name(self, name, addr, bases, sorted_mods):
        if name == "unknown":
            idx = bisect.bisect_right(bases, addr)-1
            mod_base, mod_size, mod_name = sorted_mods[idx]
            if mod_base <= addr < mod_base+mod_size:
                name = str(mod_name) + ("+%#x" % (addr - mod_base))
        else:
            # Try to undecorate
            mod, fn = name.split('!', 1)
            fn = fn.rsplit('+', 1)
            if len(fn) > 1:
                fn, offset = fn
            else:
                fn, offset = fn[0], ""
            fn = und.undname(fn)
            name = '!'.join((mod, fn))
            if offset: name = "+".join((name, offset))
        return name

    def resolve_taps(self, taps, sorted_mods, as_getter):
        bases = [m[0] for m in sorted_mods]
        matching_mods = set()
        addrs = set([a for tap in taps for a in tap[0]])
        for a in addrs:
            idx = bisect.bisect_right(bases, a)-1
            if idx == -1:
                debug.warning("No matching module for %#010x" % a)
            m_base, m_size, m_name = sorted_mods[idx]
            if not (m_base <= a < m_base+m_size):
                debug.warning("NOT %x <= %x < %x (%s)" % (m_base, a, m_base+m_size, m_name))
                continue
            matching_mods.add((m_base, m_size, m_name))

        # Fetch the PDBs for each module
        pdb_list = []
        for mod_base, mod_size, mod_name in matching_mods:
            ps_ad = as_getter(mod_base)
            if ps_ad.is_valid_address(mod_base):
                pdbname = self.get_pdb(ps_ad, mod_name, mod_base)
                if pdbname:
                    pdb_list.append((pdbname, mod_base))
                else:
                    debug.warning('No PDB found for {0} at {1:8x}'.format(mod_name, mod_base))
            else:
                debug.warning('Cannot check {0} at {1:8x}'.format(mod_name, mod_base))
        
        # Do the lookups
        lobj = lookup.Lookup(pdb_list)
        for stack, cr3 in taps:
            symbols = []
            for addr in stack:
                addr_s = lobj.lookup(addr)
                addr_s = self.format_name(addr_s, addr, bases, sorted_mods)
                symbols.append(addr_s)
            yield stack, symbols, cr3

    def calculate(self):
        if self._config.TAP_FILE is None or not os.path.exists(self._config.TAP_FILE):
            debug.error("Tap file not found")
        taplist = open(self._config.TAP_FILE, "r")

        taps = []
        for line in taplist :
            line = line.split()
            (stack, cr3) = line[:-1], line[-1]
            try:
                stack = [int(s,16) for s in stack]
                cr3 = int(cr3,16)
            except ValueError:
                debug.error("Tap file format invalid.")
            taps.append((stack, cr3))

        cr3s = set(t[1] for t in taps)
        cr3s.discard(0)

        # First do the userland ones
        addr_space = utils.load_as(self._config)
        procs = list(tasks.pslist(addr_space))
        # CR3 => proc mapping
        proc_map = {}
        for proc in procs:
            dtb = proc.Pcb.DirectoryTableBase.v()
            if dtb in cr3s: proc_map[dtb] = proc
        
        for cr3, proc in proc_map.items():
            print ">>> Working on {0}[{1}]".format(proc.ImageFileName, proc.UniqueProcessId)

            ps_ad = proc.get_process_address_space()
            proc_taps = [t for t in taps if t[1] == cr3]
            mods = list(proc.get_load_modules())
            sorted_mods = sorted([(m.DllBase.v(),m.SizeOfImage.v(),str(m.BaseDllName)) for m in mods])

            for v in self.resolve_taps(proc_taps, sorted_mods, lambda m: ps_ad):
                yield v + (proc.ImageFileName,)

        # Now kernel mode taps
        print ">>> Working on kernel taps"
        kern_taps = [t for t in taps if t[1] == 0]
        sorted_mods = sorted([(mod.DllBase.v(), mod.SizeOfImage.v(), str(mod.BaseDllName))
            for mod in modules.lsmod(addr_space)])
        bases = [m[0] for m in sorted_mods]
        for v in self.resolve_taps(kern_taps, sorted_mods, lambda m: tasks.find_space(addr_space, procs, m)):
            yield v + ("Kernel",)

    def render_text(self, outfd, data):
        for stack, symbols, cr3, procname in data:
            for addr,sym in zip(stack, symbols):
                print >>outfd, "%#010x %s" % (addr,sym),
            print >>outfd, "%#010x %s" % (cr3, procname)
